# pip install googletrans==4.0.0-rc1
# 这个库要求 httpx==0.13.3
# 但环境中的其他库可能需要更高版本的 httpx 所以为翻译任务单独创建一个环境
import googletrans
# from googletrans import Translator
from typing import Union, List
from tqdm import tqdm

# 打印支持的语言
# print(googletrans.LANGUAGES)
'''
{'af': 'afrikaans', 'sq': 'albanian', 'am': 'amharic', 'ar': 'arabic', 'hy': 'armenian', 'az': 'azerbaijani', 'eu': 'basque', 
'be': 'belarusian', 'bn': 'bengali', 'bs': 'bosnian', 'bg': 'bulgarian', 'ca': 'catalan', 'ceb': 'cebuano', 'ny': 'chichewa', 
'zh-cn': 'chinese (simplified)', 'zh-tw': 'chinese (traditional)', 'co': 'corsican', 'hr': 'croatian', 'cs': 'czech', 'da': 'danish', 
'nl': 'dutch', 'en': 'english', 'eo': 'esperanto', 'et': 'estonian', 'tl': 'filipino', 'fi': 'finnish', 'fr': 'french', 'fy': 'frisian', 
'gl': 'galician', 'ka': 'georgian', 'de': 'german', 'el': 'greek', 'gu': 'gujarati', 'ht': 'haitian creole', 'ha': 'hausa', 
'haw': 'hawaiian', 'iw': 'hebrew', 'he': 'hebrew', 'hi': 'hindi', 'hmn': 'hmong', 'hu': 'hungarian', 'is': 'icelandic', 'ig': 'igbo', 
'id': 'indonesian', 'ga': 'irish', 'it': 'italian', 'ja': 'japanese', 'jw': 'javanese', 'kn': 'kannada', 'kk': 'kazakh', 'km': 'khmer', 
'ko': 'korean', 'ku': 'kurdish (kurmanji)', 'ky': 'kyrgyz', 'lo': 'lao', 'la': 'latin', 'lv': 'latvian', 'lt': 'lithuanian', 
'lb': 'luxembourgish', 'mk': 'macedonian', 'mg': 'malagasy', 'ms': 'malay', 'ml': 'malayalam', 'mt': 'maltese', 'mi': 'maori', 
'mr': 'marathi', 'mn': 'mongolian', 'my': 'myanmar (burmese)', 'ne': 'nepali', 'no': 'norwegian', 'or': 'odia', 'ps': 'pashto', 
'fa': 'persian', 'pl': 'polish', 'pt': 'portuguese', 'pa': 'punjabi', 'ro': 'romanian', 'ru': 'russian', 'sm': 'samoan', 'gd': 'scots gaelic', 
'sr': 'serbian', 'st': 'sesotho', 'sn': 'shona', 'sd': 'sindhi', 'si': 'sinhala', 'sk': 'slovak', 'sl': 'slovenian', 'so': 'somali', 
'es': 'spanish', 'su': 'sundanese', 'sw': 'swahili', 'sv': 'swedish', 'tg': 'tajik', 'ta': 'tamil', 'te': 'telugu', 'th': 'thai', 
'tr': 'turkish', 'uk': 'ukrainian', 'ur': 'urdu', 'ug': 'uyghur', 'uz': 'uzbek', 'vi': 'vietnamese', 'cy': 'welsh', 'xh': 'xhosa', 
'yi': 'yiddish', 'yo': 'yoruba', 'zu': 'zulu'}
'''

# 设置Google翻译服务地址
# translator = Translator(service_urls=[
#       'translate.google.com'
# ])
# 不设置的话 应该有默认值 translate.google.com
translator = googletrans.Translator()

# 测试一下
# translation = translator.translate('A woman with a black shirt and tan apron is standing behind a counter in a restaurant .', dest='zh-cn')
# print(translation.text)

def google_translate(text: Union[str, List[str]], dest: str='zh-cn', src: str='auto') -> Union[str, List[str]]:
    translator = googletrans.Translator()

    if isinstance(text, str):
        try:
            translation = translator.translate(text=text, dest=dest, src=src)
            translation_text = translation.text
        except Exception as e:
            print(f"Error translating text: {e}")
            translation_text = "error error error la"

    # elif isinstance(text, list) and isinstance(text[0], str):
    #     translation_text = []
    #     try:
    #         has_error = False
    #         for item in tqdm(text, total=len(text)):
    #             translation = translator.translate(text=item, dest=dest, src=src)
    #             translation_text.append(translation.text)
    #     except Exception as e:
    #         print(f"Error translating '{item}': {e}")
    #         translation_text.append("errorerrorerror")
    #         has_error = True

    #     if has_error:
    #         print("所有文本已经翻译完成 过程中出现了翻译错误 请检查值为 'errorerrorerror' 的数据")
    #     else:
    #         print("所有文本已经翻译完成 没有出现错误")

    elif isinstance(text, list) and isinstance(text[0], str):
        translation_text = []

        has_error = False
        has_error_num = 0

        for item in tqdm(text, total=len(text), desc="Translating"):
        # for item in tqdm(text, total=len(text)):
            try:
                translation = translator.translate(text=item, dest=dest)
                translation_text.append(translation.text)

                # 如果没有错误 重置错误次数计数器 
                if has_error_num <= 5:
                    has_error_num = 0

            except Exception as e:
                print(f"Error translating '{item}': {e}")
                translation_text.append("error error error la")

                if not has_error:
                    has_error = True
                    has_error_num = has_error_num + 1
                else:
                    has_error_num += 1

            # 如果出错的次数太多了 可能是 time out 了 就跳出循环
            if has_error_num > 5:
                break
        
        if has_error and has_error_num > 5:
            print("出现了错误 只翻译了部分文本")
        elif has_error and has_error_num <= 5:
            print("所有文本已经翻译完成 过程中出现了翻译错误 请检查值为 'error error error la' 的数据")
        else:
            print("所有文本已经翻译完成 没有出现错误")

    
    else:
        raise ValueError("请检查翻译文本的数据类型是否为 str 或者 list[str] ")

    return translation_text
