# TensorlayerX framework deeplearning code convert from 

import types, math, os, gc

import tensorlayerx as tlx


import numpy as np
import h5py

MyModule = tlx.nn.Module

def __nop(ob):
    return ob
MyFunction = __nop

RWKV_RESCALE_LAYER = 6 # set x = x/2 every X layer (to avoid FP16 overflow)

############################################################################################################

class RWKV_RNN(MyModule):
    def __init__(self, args):
        super().__init__()

        self.args = args
        if args.FLOAT_MODE == 'fp32':
            self.FLOAT_MODE = tlx.float
        elif args.FLOAT_MODE == 'fp16':
            self.FLOAT_MODE = tlx.float16
        elif args.FLOAT_MODE == 'bf16':
            self.FLOAT_MODE = tlx.bfloat16
        self.RUN_DEVICE = args.RUN_DEVICE

        self.load(args)

    def load(self,args):
        # Load weights from .h5
        with h5py.File(args.MODEL_NAME + '.h5', 'r') as f:
            w = {}
            for k in f.keys():
                w[k] = tlx.convert_to_tensor(f[k][()], dtype=self.FLOAT_MODE)

        args.n_embd = w['emb.weight'].shape[1]
        args.n_layer = 0
        keys = list(w.keys()) # refine weights and send to correct device
        print_need_newline = False
        for x in keys:
            w[x].requires_grad = False
            if x == 'emb.weight' or 'ln0' in x:
                continue
            
            block_id = int(x.split('.')[1]) if ('blocks.' in x) else 0
            args.n_layer = max(args.n_layer, block_id+1)
            
            if '.time_' in x:
                w[x] = w[x].squeeze()
            if 'key.weight' in x or 'value.weight' in x or 'receptance.weight' in x or 'output.weight' in x:
                w[x] = w[x].t()
            
            if '.time_decay' in x:
                w[x] = w[x].float()
                w[x] = -tlx.exp(w[x])
            elif '.time_first' in x:
                w[x] = w[x].float()
            else:
                w[x] = w[x].to(dtype=self.FLOAT_MODE)

            if args.FLOAT_MODE == 'fp16':
                if 'att.output.weight' in x:
                    w[x] = w[x] / (2 ** int(block_id // RWKV_RESCALE_LAYER))
                if 'ffn.value.weight' in x:
                    w[x] = w[x] / (2 ** int(block_id // RWKV_RESCALE_LAYER))
            
            if args.RUN_DEVICE == 'cuda':
                w[x] = w[x].cuda()

            shape = w[x].shape
            shape = [i for i in shape if i != 1]
            if len(shape) > 1:
                shape = f"  {str(shape[0]).rjust(5)} {str(shape[1]).rjust(5)}"
            else:
                shape = f"  {str(shape[0]).rjust(5)}      "
            if block_id == 0:
                if print_need_newline:
                    print('\n', end = '')
                    print_need_newline = False
                print(x.ljust(32), str(w[x].dtype).replace('torch.', '').ljust(10), w[x].device, shape)
            else:
                print_need_newline = True
                print('.', end = '', flush = True)
        print(f'\nn_layer {args.n_layer} n_embd {args.n_embd} ctx_len {args.ctx_len}')

        keys = list(w.keys()) # store weights in self.w
        self.w = types.SimpleNamespace()
        for x in keys:
            xx = x.split('.')
            here = self.w
            for i in range(len(xx)):
                if xx[i].isdigit():
                    ii = int(xx[i])
                    if ii not in here:
                        here[ii] = types.SimpleNamespace()
                    here = here[ii]
                else:
                    if i == len(xx) - 1:
                        setattr(here, xx[i], w[x])
                    elif not hasattr(here, xx[i]):
                        if xx[i+1].isdigit():
                            setattr(here, xx[i], {})
                        else:
                            setattr(here, xx[i], types.SimpleNamespace())
                    here = getattr(here, xx[i])

        try:
            x = self.LN(self.w.emb.weight, self.w.blocks[0].ln0)
        except:
            input_shape = self.w.emb.weight.float().shape
            x = tlx.ops.layernorm((self.args.n_embd,), self.w.blocks[0].ln0.weight.float(), self.w.blocks[0].ln0.bias.float(),1e-5, input_shape)(self.w.emb.weight.float())
            # x = F.layer_norm(self.w.emb.weight.float(), )
        self.w.emb.weight = x.to(dtype=self.FLOAT_MODE)

        self.eval()
        gc.collect()

    def LN(self, x, w):
        normalized_shape = (self.args.n_embd,)
        gama = w.weight
        beta = w.bias
        eps = 1e-5
        input_shape = x.shape
        return tlx.ops.layernorm(normalized_shape, gama, beta, eps, input_shape)(x)

    # state[] 0=ffn_xx 1=att_xx 2=att_aa 3=att_bb 4=att_pp

    @MyFunction
    def FF_one(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
        xx = state[5*i+0].to(dtype=self.FLOAT_MODE)
        xk = x * time_mix_k + xx * (1 - time_mix_k)
        xr = x * time_mix_r + xx * (1 - time_mix_r)
        state[5*i+0] = x.float()

        r = tlx.sigmoid(xr @ rw)
        k = tlx.square(tlx.relu(xk @ kw))
        kv = k @ vw
        return r * kv

    @MyFunction
    def FF_seq(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
        xx = tlx.concat((state[5*i+0].to(dtype=self.FLOAT_MODE).unsqueeze(0), x[:-1,:]))
        xk = x * time_mix_k + xx * (1 - time_mix_k)
        xr = x * time_mix_r + xx * (1 - time_mix_r)
        state[5*i+0] = x[-1,:].float()

        r = tlx.sigmoid(xr @ rw)
        k = tlx.square(tlx.relu(xk @ kw))
        kv = k @ vw
        return r * kv

    @MyFunction
    def SA_one(self, x, state, i:int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
        xx = state[5*i+1].to(dtype=self.FLOAT_MODE)
        xk = x * time_mix_k + xx * (1 - time_mix_k)
        xv = x * time_mix_v + xx * (1 - time_mix_v)
        xr = x * time_mix_r + xx * (1 - time_mix_r)
        state[5*i+1] = x.float()

        r = tlx.sigmoid(xr @ rw)
        k = (xk @ kw).float()
        v = (xv @ vw).float()

        aa = state[5*i+2]
        bb = state[5*i+3]
        pp = state[5*i+4]
        ww = time_first + k
        p = tlx.maximum(pp, ww)
        e1 = tlx.exp(pp - p)
        e2 = tlx.exp(ww - p)
        a = e1 * aa + e2 * v
        b = e1 * bb + e2
        ww = pp + time_decay
        p = tlx.maximum(ww, k)
        e1 = tlx.exp(ww - p)
        e2 = tlx.exp(k - p)
        state[5*i+2] = e1 * aa + e2 * v
        state[5*i+3] = e1 * bb + e2
        state[5*i+4] = p
        wkv = (a / b).to(dtype=self.FLOAT_MODE)
        return (r * wkv) @ ow

    @MyFunction
    def SA_seq(self, x, state, i:int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
        xx = tlx.concat((state[5*i+1].to(dtype=self.FLOAT_MODE).unsqueeze(0), x[:-1,:]))
        xk = x * time_mix_k + xx * (1 - time_mix_k)
        xv = x * time_mix_v + xx * (1 - time_mix_v)
        xr = x * time_mix_r + xx * (1 - time_mix_r)
        state[5*i+1] = x[-1,:].float()

        r = tlx.sigmoid(xr @ rw)
        k = (xk @ kw).float()
        v = (xv @ vw).float()

        aa = state[5*i+2]
        bb = state[5*i+3]
        pp = state[5*i+4]
        T = x.shape[0]
        for t in range(T):
            ww = time_first + k[t]
            p = tlx.maximum(pp, ww)
            e1 = tlx.exp(pp - p)
            e2 = tlx.exp(ww - p)
            a = e1 * aa + e2 * v[t]
            b = e1 * bb + e2
            ww = pp + time_decay
            p = tlx.maximum(ww, k[t])
            e1 = tlx.exp(ww - p)
            e2 = tlx.exp(k[t] - p)
            if t != T - 1:
                aa = e1 * aa + e2 * v[t]
                bb = e1 * bb + e2
                pp = p
            else:
                state[5*i+2] = e1 * aa + e2 * v[t]
                state[5*i+3] = e1 * bb + e2
                state[5*i+4] = p
            xx[t] = (a / b).to(dtype=self.FLOAT_MODE)
        return (r * xx) @ ow

    def forward(self, tokens, state, preprocess_only = False):
        w = self.w
        args = self.args

        seq_mode = len(tokens) > 1

        x = w.emb.weight[tokens] if seq_mode else w.emb.weight[tokens[-1]]
        if self.RUN_DEVICE == 'cuda':
            x = x.cuda()

        if state == None:
            state = tlx.zeros([args.n_layer * 5, args.n_embd], device=self.RUN_DEVICE)
            for i in range(args.n_layer):
                state[5*i+4] -= 1e30

        SA = self.SA_seq if seq_mode else self.SA_one
        FF = self.FF_seq if seq_mode else self.FF_one

        for i in range(args.n_layer):
            ww = w.blocks[i].att
            x = x + SA(self.LN(x, w.blocks[i].ln1), state, i, 
                ww.time_mix_k, ww.time_mix_v, ww.time_mix_r, ww.time_first, ww.time_decay, 
                ww.key.weight, ww.value.weight, ww.receptance.weight, ww.output.weight)
            
            ww = w.blocks[i].ffn
            x = x + FF(self.LN(x, w.blocks[i].ln2), state, i, 
                ww.time_mix_k, ww.time_mix_r, 
                ww.key.weight, ww.value.weight, ww.receptance.weight)
            
            if args.FLOAT_MODE == 'fp16':
                if (i+1) % RWKV_RESCALE_LAYER == 0:
                    x = x / 2

        if preprocess_only:
            return state

        x = self.LN(x[-1,:], w.ln_out) if seq_mode else self.LN(x, w.ln_out)
        x = w.head.weight @ x

        return x.float(), state

