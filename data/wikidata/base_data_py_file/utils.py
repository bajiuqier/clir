import string


def is_english_or_chinese(text):
    '''
    检查字符串是中文还是英文还是混合两种语言
    全为中文返回 字符串 "Chinese"
    全为英文返回 字符串 "English"
    混合两种语言返回 字符串 "Mixed"
    '''

    # 删除所有空格符
    text = text.replace(' ', '')
    # 去除标点符号
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 初始化计数器
    english_count = 0
    chinese_count = 0
    
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            chinese_count += 1
        elif '\u0041' <= char <= '\u005a' or '\u0061' <= char <= '\u007a':
            english_count += 1
    
    if chinese_count > 0 and english_count == 0:
        return "Chinese"
    elif english_count > 0 and chinese_count == 0:
        return "English"
    else:
        return "Mixed"
    
