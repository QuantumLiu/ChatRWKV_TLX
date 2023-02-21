########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, copy, types, gc, sys
os.environ["TL_BACKEND"] = "torch"

import numpy as np
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass
np.set_printoptions(precision=4, suppress=True, linewidth=200)
args = types.SimpleNamespace()

print('\n\nChatRWKV project: https://github.com/BlinkDL/ChatRWKV')

########################################################################################################

args.RUN_DEVICE = "cuda"  # cuda // cpu
# fp16 (good for GPU, does NOT support CPU) // fp32 (good for CPU) // bf16 (worse accuracy, supports CPU)
args.FLOAT_MODE = "fp16"

os.environ["RWKV_JIT_ON"] = '1' # '1' or '0', please use torch 1.13+ and benchmark speed

CHAT_LANG = 'English' # English // Chinese // more to come

QA_PROMPT = False # True: Q & A prompt // False: User & Bot prompt
# 中文问答设置QA_PROMPT=True（只能问答，问答效果更好，但不能闲聊） 中文聊天设置QA_PROMPT=False（可以闲聊，但需要大模型才适合闲聊）

# Download RWKV-4 models from https://huggingface.co/BlinkDL

if CHAT_LANG == 'English':
    # args.MODEL_NAME = '/mnt/data/RWKV-4-Pile-14B-20230204-7324'
    args.MODEL_NAME = '/mnt/data/RWKV-4-Pile-7B-20221115-8047'
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-3b/RWKV-4-Pile-3B-20221110-ctx4096'
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-3b/RWKV-4-Pile-3B-Instruct-test1-20230124'
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040'
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-430m/RWKV-4-Pile-430M-20220808-8066'
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-169m/RWKV-4-Pile-169M-20220807-8023'
    # args.MODEL_NAME = '/fsx/BlinkDL/CODE/_PUBLIC_/RWKV-LM/RWKV-v4neo/7-run1z/rwkv-340'
    # args.MODEL_NAME = '/fsx/BlinkDL/CODE/_PUBLIC_/RWKV-LM/RWKV-v4neo/14b-run1/rwkv-6210'

elif CHAT_LANG == 'Chinese':
    args.MODEL_NAME = '/mnt/data/RWKV-4-Pile-7B-EngChn-test4-20230116'
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-3b/RWKV-4-Pile-3B-EngChn-test4-20230115'
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-EngChn-test4-20230115'
    # args.MODEL_NAME = '/fsx/BlinkDL/CODE/_PUBLIC_/RWKV-LM/RWKV-v4neo/7-run1z/rwkv-490'
    # args.MODEL_NAME = '/fsx/BlinkDL/CODE/_PUBLIC_/RWKV-LM/RWKV-v4neo/1.5-run1z/rwkv-415'

args.ctx_len = 1024

CHAT_LEN_SHORT = 40
CHAT_LEN_LONG = 150
FREE_GEN_LEN = 200

GEN_TEMP = 1.0
GEN_TOP_P = 0.85

AVOID_REPEAT = '，。：？！'

########################################################################################################

os.environ["RWKV_RUN_DEVICE"] = args.RUN_DEVICE
print(f'\nLoading ChatRWKV - {CHAT_LANG} - {args.RUN_DEVICE} - {args.FLOAT_MODE} - QA_PROMPT {QA_PROMPT}')
# import torch
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.allow_tf32 = True
# torch.backends.cuda.matmul.allow_tf32 = True
from src_tlx.model_run import RWKV_RNN
from src_tlx.utils import TOKENIZER
tokenizer = TOKENIZER("20B_tokenizer.json")

args.vocab_size = 50277
args.head_qk = 0
args.pre_ffn = 0
args.grad_cp = 0
args.my_pos_emb = 0
MODEL_NAME = args.MODEL_NAME

if CHAT_LANG == 'English':
    interface = ":"

    if QA_PROMPT:
        user = "Q"
        bot = "A"
        intro = f'The following is a verbose and detailed Q & A conversation of factual information.'
    else:
        user = "User"
        bot = "Bot"
        intro = f'The following is a verbose and detailed conversation between an AI assistant called {bot}, and a human user called {user}. {bot} is intelligent, knowledgeable, wise and polite.'
    
    init_prompt = f'''
{intro}

{user}{interface} french revolution what year

{bot}{interface} The French Revolution started in 1789, and lasted 10 years until 1799.

{user}{interface} 3+5=?

{bot}{interface} The answer is 8.

{user}{interface} guess i marry who ?

{bot}{interface} Only if you tell me more about yourself - what are your interests?

{user}{interface} solve for a: 9-a=2

{bot}{interface} The answer is a = 7, because 9 - 7 = 2.

{user}{interface} wat is lhc

{bot}{interface} LHC is a high-energy particle collider, built by CERN, and completed in 2008. They used it to confirm the existence of the Higgs boson in 2012.

'''
    HELP_MSG = '''Commands:
say something --> chat with bot. use \\n for new line.
+ --> alternate chat reply
+reset --> reset chat

+gen YOUR PROMPT --> free generation with any prompt. use \\n for new line.
+qa YOUR QUESTION --> free generation - ask any question (just ask the question). use \\n for new line.
+++ --> continue last free generation (only for +gen / +qa)
++ --> retry last free generation (only for +gen / +qa)

Now talk with the bot and enjoy. Remember to +reset periodically to clean up the bot's memory. Use RWKV-4 14B for best results.
This is not instruct-tuned for conversation yet, so don't expect good quality. Better use +gen for free generation.

Prompt is VERY important. Try all prompts on https://github.com/BlinkDL/ChatRWKV first.
'''
elif CHAT_LANG == 'Chinese':
    interface = ":"
    if QA_PROMPT:
        user = "Q"
        bot = "A"
        init_prompt = f'''
Expert Questions & Helpful Answers

Ask Research Experts

'''
    else:
        user = "User"
        bot = "Bot"
        init_prompt = f'''
The following is a verbose and detailed conversation between an AI assistant called {bot}, and a human user called {user}. {bot} is intelligent, knowledgeable, wise and polite.

{user}{interface} wat is lhc

{bot}{interface} LHC is a high-energy particle collider, built by CERN, and completed in 2008. They used it to confirm the existence of the Higgs boson in 2012.

{user}{interface} 企鹅会飞吗

{bot}{interface} 企鹅是不会飞的。它们的翅膀主要用于游泳和平衡，而不是飞行。

'''
    HELP_MSG = '''指令:

直接输入内容 --> 和机器人聊天（建议问机器人问题），用\\n代表换行
+ --> 让机器人换个回答
+reset --> 重置对话

+gen 某某内容 --> 续写任何中英文内容，用\\n代表换行
+qa 某某问题 --> 问独立的问题（忽略上下文），用\\n代表换行
+qq 某某问题 --> 问独立的问题（忽略上下文），且敞开想象力，用\\n代表换行
+++ --> 继续 +gen / +qa / +qq 的回答
++ --> 换个 +gen / +qa / +qq 的回答

作者：彭博 请关注我的知乎: https://zhuanlan.zhihu.com/p/603840957

如果喜欢，欢迎看我们的优质护眼灯: https://withablink.taobao.com

现在可以输入内容和机器人聊天（注意它不大懂中文，它更懂英文）。请经常使用 +reset 重置机器人记忆。
目前没有“重复惩罚”，所以机器人有时会重复，此时必须使用 + 换成正常回答，以免污染电脑记忆。
注意：和上下文无关的独立问题，必须用 +qa 或 +qq 问，以免污染电脑记忆。

请先试下列咒语，理解咒语的写法！咒语至关重要。
+gen \\n活动出席发言稿：\\n大家好，
+gen \\n怎样创立一家快速盈利的AI公司：\\n1.
+gen 二向箔是一种超级武器，它的原理是
+gen \\nimport torch
【下面这些多试几次】
+qq 请以《我的驴》为题写一篇作文
+qq 请以《企鹅》为题写一首诗歌
+qq 请设定一个奇幻世界，告诉我详细的世界设定。
'''

# Load Model

print(f'Loading model - {MODEL_NAME}')
model = RWKV_RNN(args)

model_tokens = []
model_state = None

AVOID_REPEAT_TOKENS = []
for i in AVOID_REPEAT:
    dd = tokenizer.encode(i)
    assert len(dd) == 1
    AVOID_REPEAT_TOKENS += dd

########################################################################################################

def run_rnn(tokens, newline_adj = 0):
    global model_tokens, model_state

    tokens = [int(x) for x in tokens]
    model_tokens += tokens
    out, model_state = model.forward(tokens, model_state)

    # print(f'### model ###\n{tokens}\n[{tokenizer.decode(model_tokens)}]')

    out[0] = -999999999  # disable <|endoftext|>
    out[187] += newline_adj # adjust \n probability
    # if newline_adj > 0:
    #     out[15] += newline_adj / 2 # '.'
    if model_tokens[-1] in AVOID_REPEAT_TOKENS:
        out[model_tokens[-1]] = -999999999
    return out

all_state = {}
def save_all_stat(srv, name, last_out):
    n = f'{name}_{srv}'
    all_state[n] = {}
    all_state[n]['out'] = last_out
    all_state[n]['rnn'] = copy.deepcopy(model_state)
    all_state[n]['token'] = copy.deepcopy(model_tokens)

def load_all_stat(srv, name):
    global model_tokens, model_state
    n = f'{name}_{srv}'
    model_state = copy.deepcopy(all_state[n]['rnn'])
    model_tokens = copy.deepcopy(all_state[n]['token'])
    return all_state[n]['out']

########################################################################################################

# Run inference
print(f'\nRun prompt...')

out = run_rnn(tokenizer.encode(init_prompt))
save_all_stat('', 'chat_init', out)
gc.collect()
# torch.cuda.empty_cache()

srv_list = ['dummy_server']
for s in srv_list:
    save_all_stat(s, 'chat', out)

print(f'### prompt ###\n[{tokenizer.decode(model_tokens)}]\n')

def reply_msg(msg):
    print(f'{bot}{interface} {msg}\n')

def on_message(message):
    global model_tokens, model_state

    srv = 'dummy_server'

    msg = message.replace('\\n','\n').strip()
    # if len(msg) > 1000:
    #     reply_msg('your message is too long (max 1000 tokens)')
    #     return

    x_temp = GEN_TEMP
    x_top_p = GEN_TOP_P
    if ("-temp=" in msg):
        x_temp = float(msg.split("-temp=")[1].split(" ")[0])
        msg = msg.replace("-temp="+f'{x_temp:g}', "")
        # print(f"temp: {x_temp}")
    if ("-top_p=" in msg):
        x_top_p = float(msg.split("-top_p=")[1].split(" ")[0])
        msg = msg.replace("-top_p="+f'{x_top_p:g}', "")
        # print(f"top_p: {x_top_p}")
    if x_temp <= 0.2:
        x_temp = 0.2
    if x_temp >= 5:
        x_temp = 5
    if x_top_p <= 0:
        x_top_p = 0
    
    if msg == '+reset':
        out = load_all_stat('', 'chat_init')
        save_all_stat(srv, 'chat', out)
        reply_msg("Chat reset.")
        return

    elif msg[:5].lower() == '+gen ' or msg[:4].lower() == '+qa ' or msg[:4].lower() == '+qq ' or msg.lower() == '+++' or msg.lower() == '++':

        if msg[:5].lower() == '+gen ':
            new = '\n' + msg[5:].strip()
            # print(f'### prompt ###\n[{new}]')
            model_state = None
            model_tokens = []
            out = run_rnn(tokenizer.encode(new))
            save_all_stat(srv, 'gen_0', out)

        elif msg[:4].lower() == '+qq ':
            new = '\nQ: ' + msg[4:].strip() + '\nA:'
            # print(f'### prompt ###\n[{new}]')
            model_state = None
            model_tokens = []
            out = run_rnn(tokenizer.encode(new))
            save_all_stat(srv, 'gen_0', out)

        elif msg[:4].lower() == '+qa ':
            out = load_all_stat('', 'chat_init')

            real_msg = msg[4:].strip()
            new = f"{user}{interface} {real_msg}\n\n{bot}{interface}"
            # print(f'### qa ###\n[{new}]')
            
            out = run_rnn(tokenizer.encode(new))
            save_all_stat(srv, 'gen_0', out)

        elif msg.lower() == '+++':
            try:
                out = load_all_stat(srv, 'gen_1')
                save_all_stat(srv, 'gen_0', out)
            except:
                return

        elif msg.lower() == '++':
            try:
                out = load_all_stat(srv, 'gen_0')
            except:
                return

        begin = len(model_tokens)
        out_last = begin
        for i in range(FREE_GEN_LEN+100):
            token = tokenizer.sample_logits(
                out,
                model_tokens,
                args.ctx_len,
                temperature=x_temp,
                top_p=x_top_p,
            )
            if msg[:4].lower() == '+qa ':# or msg[:4].lower() == '+qq ':
                out = run_rnn([token], newline_adj=-2)
            else:
                out = run_rnn([token])
            
            xxx = tokenizer.decode(model_tokens[out_last:])
            if '\ufffd' not in xxx: # avoid utf-8 display issues
                print(xxx, end='', flush=True)
                out_last = begin + i + 1
                if i >= FREE_GEN_LEN:
                    break
        print('\n')
        # send_msg = tokenizer.decode(model_tokens[begin:]).strip()
        # print(f'### send ###\n[{send_msg}]')
        # reply_msg(send_msg)
        save_all_stat(srv, 'gen_1', out)

    else:
        if msg.lower() == '+':
            try:
                out = load_all_stat(srv, 'chat_pre')
            except:
                return
        else:
            out = load_all_stat(srv, 'chat')
            new = f"{user}{interface} {msg}\n\n{bot}{interface}"
            # print(f'### add ###\n[{new}]')
            out = run_rnn(tokenizer.encode(new), newline_adj=-999999999)
            save_all_stat(srv, 'chat_pre', out)

        begin = len(model_tokens)
        out_last = begin
        print(f'{bot}{interface}', end='', flush=True)
        for i in range(999):
            if i <= 0:
                newline_adj = -999999999
            elif i <= CHAT_LEN_SHORT:
                newline_adj = (i - CHAT_LEN_SHORT) / 10
            elif i <= CHAT_LEN_LONG:
                newline_adj = 0
            else:
                newline_adj = (i - CHAT_LEN_LONG) * 0.25 # MUST END THE GENERATION
            token = tokenizer.sample_logits(
                out,
                model_tokens,
                args.ctx_len,
                temperature=x_temp,
                top_p=x_top_p,
            )
            out = run_rnn([token], newline_adj=newline_adj)

            xxx = tokenizer.decode(model_tokens[out_last:])
            if '\ufffd' not in xxx: # avoid utf-8 display issues
                print(xxx, end='', flush=True)
                out_last = begin + i + 1
            
            send_msg = tokenizer.decode(model_tokens[begin:])
            if '\n\n' in send_msg:
                send_msg = send_msg.strip()
                break
            
            # send_msg = tokenizer.decode(model_tokens[begin:]).strip()
            # if send_msg.endswith(f'{user}{interface}'): # warning: needs to fix state too !!!
            #     send_msg = send_msg[:-len(f'{user}{interface}')].strip()
            #     break
            # if send_msg.endswith(f'{bot}{interface}'):
            #     send_msg = send_msg[:-len(f'{bot}{interface}')].strip()
            #     break

        # print(f'{model_tokens}')
        # print(f'[{tokenizer.decode(model_tokens)}]')

        # print(f'### send ###\n[{send_msg}]')
        # reply_msg(send_msg)
        save_all_stat(srv, 'chat', out)

print(HELP_MSG)

while True:
    msg = input(f'{user}{interface} ')
    if len(msg.strip()) > 0:
        on_message(msg)
    else:
        print('Error: please say something')
