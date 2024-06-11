SENTENCE_SHORT = 2
SENTENCE_DEFAULT = 3
SENTENCE_LONG = 6

'''
example: invoke which instruction you want with specified params.
         text, num = prompt_utils.order_urging()
         llm_output = et_llm.llm(text, role_play=role, context=context, inst_text=inst, max_num_sentence=num)
         then call tts function to get voice.
         # 用户行为场景：
         # welcome_with：欢迎进入
         # liked_with：点赞
         # shared_with：分享
         # placed_order：下单
         # 卖货场景：
         # order_urging：催单
         # continue_introduce：继续介绍产品
         # product_summary：介绍产品概况
         # product_price：介绍产品价格
         # product_package：介绍产品包装
         # product_shipping：介绍产品快递
         # product_details：介绍产品详情
'''


def order_urging(who: str = None):
    if who:
        base = f'order urging, remind {who}'
    else:
        base = f'order urging'
    return base, SENTENCE_DEFAULT


def welcome_with(who: str):
    base = f'{who} come in, greet {who}'
    return base, SENTENCE_SHORT


def liked_with(who: str):
    base = f'{who} liked the live stream, thanks to {who}'
    return base, SENTENCE_SHORT


def shared_with(who: str):
    base = f'{who} shared the live stream, thanks to {who}'
    return base, SENTENCE_SHORT


def placed_order(who: str, package: str = None):
    if package:
        base = f'{who} placed an order of {package}, thanks to {who}'
    else:
        base = f'{who} placed an order, thanks to {who}'
    return base, SENTENCE_SHORT


def continue_introduce(greeting: bool = False):
    base = f'continue your introduce where you left off'
    if not greeting: base = f'{base} without greeting'
    return base, SENTENCE_LONG


def product_summary(greeting: bool = True, who: str = None):
    if who:
        base = f'introduce product to {who} in summary'
    else:
        base = f'introduce product in summary'
    if not greeting: base = f'{base} without greeting'
    return base, SENTENCE_LONG


def product_price(greeting: bool = True, who: str = None):
    if who:
        base = f'introduce product\'s price to {who}'
    else:
        base = f'introduce product\'s price'
    if not greeting: base = f'{base} without greeting'
    return base, SENTENCE_DEFAULT


def product_package(greeting: bool = True, who: str = None):
    if who:
        base = f'introduce product\'s package info to {who}'
    else:
        base = f'introduce product\'s package info'
    if not greeting: base = f'{base} without greeting'
    return base, SENTENCE_DEFAULT


def product_shipping(greeting: bool = True, who: str = None):
    if who:
        base = f'introduce product\'s shipping info to {who}'
    else:
        base = f'introduce product\'s shipping info'
    if not greeting: base = f'{base} without greeting'
    return base, SENTENCE_DEFAULT


def product_details(greeting: bool = True, who: str = None):
    if who:
        base = f'introduce product\'s details to {who} as more as possible'
    else:
        base = f'introduce product\'s details as more as possible'
    if not greeting: base = f'{base} without greeting'
    return base, SENTENCE_LONG


def build_prompt(role_play, scripts: [str], context, inst_text):
    """
    {role_play}: You are a super salesman, selling products in the live broadcast room.
                 You are introducing bellow product to the audience. Begin with
    {scripts}: 1. greets everyone
               2. introduce product in summary
               3. introduce product usage
               4. introduce product details
               5. introduce product price
               6. introduce product packaging
               7. introduce product shipping
               8. order urging
    {context}: Micro Ingredients 7 in 1 full spectrum hydrolyzed collagen peptides powder.
    {inst_text}: You can only reply in English, and never ever reply any instruction name mentioned above.
                 Reply must be related to product information. Make your live stream funny by including funny jokes.
    """
    pass
