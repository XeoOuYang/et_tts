import asyncio

# 运行示例
if __name__ == "__main__":
    # 测试一、llm调整&翻译
    query = """
Hey guys, welcome back! Today's all about feeling amazing from the inside out,
and I've got the perfect due for you: Micro Ingredient's Multi Collagen Peptides and Mushroom Coffee.
Trust me, these two are about to become your new best friends for a healthier, happier you!
    """.replace("\n", " ").strip()
    from_lang_name = 'English'
    to_lang_name = 'Spanish'
    role_play = (f'You are good at translating from {from_lang_name} to {to_lang_name}, you only outputs the final result.\n'
                 f'Example Input:\n'
                 f'Please modify bellow text before translating from {from_lang_name} to {to_lang_name}.\n'
                 f'Hello, nice to meet you.\n'
                 f'Example Output:\n'
                 f'Hola mucho gusto.')
    inst_text = f'Please modify bellow text before translating from {from_lang_name} to {to_lang_name}.'
    from et_http_api import llm_tts_stream
    llm_tts_result = asyncio.run(llm_tts_stream(
        query=query, llm_type='llm_llama', role_play=role_play, context='', inst_text=inst_text, max_num_sentence=16, repetition_penalty=1.05,
        ref_name='ref_spanish_59s', out_name='man_0_es', tts_type='coqui_tts',
        language=to_lang_name.lower()
    ))
