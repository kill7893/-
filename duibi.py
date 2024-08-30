import json
import difflib
#对比两个文本内容的相似度
def calculate_similarity(text1, text2):
    seq = difflib.SequenceMatcher(None, text1, text2)
    similarity = seq.ratio()
    return similarity
flag = 0
#读取文本内容
with open('system_results.txt', 'r', encoding='utf-8') as f:
    content = f.read()
#使用json模块来解析这个json格式的字符串
data=json.loads(content)
#只提取文本内容信息，并在提取的过程进行违规检测对比
for d in data:
    #提取一串transcription后的文本内容
    str=d['transcription']
    #读取违规词库中文本的关键词，并保存在text1中
    with open('违规词库.txt', 'r', encoding='utf-8') as file:
        for line in file:
            text = line.strip()
            length = len(text)
            #如果待对比文本内容长度小于关键词长度，跳到下一次对比
            if len(str) < length :
                continue
            # 1.处理str将其分成快，按照违规关键词的长度进行分块：滑动窗口算法
            i = 0
            while i <= len(str) - length:
                sub_str = str[i:i + length]
                similarity = calculate_similarity(text,sub_str)
                if similarity >= 0.8 :
                    print("截取的文本内容块：" + sub_str)
                    print("违规关键词：" + text)
                    print(f"对比相似度: {similarity}")
                    flag = flag + 1
                i += 1
if flag > 0 :
    print("此广告内容违规！")





