from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from hanspell import spell_checker
import re

def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model

def load_tokenizer(tokenizer_path):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    return tokenizer

def generate_text(sequence, max_length):
    model_path = 'C:/Workspace/python/빅데이터 지능형서비스 개발 팀프로젝트/Final Project/Data/KoGPT2_Data/Model_Data/수필'
    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    ids = tokenizer.encode(f'{sequence},', return_tensors = 'pt')
    final_outputs = model.generate(
        ids,
        do_sample = True,
        max_length = max_length,
        pad_token_id = model.config.pad_token_id,
        tok_k = 5, # 가장 높은 확률을 지닌 n개의 단어수 중에서 추출
        top_p = 0.90, # 누적확률이 n%인 단어까지 포함하여 그 중에서 추출
        no_repeat_ngram_size = 3,
        repetition_penalty = 1.5, # 단어사용 반복에 대한 패널티 부여
    )
    return tokenizer.decode(final_outputs[0])

def spell_check(sequence):
    result = spell_checker.check(sequence)
    checked_sequence = result.checked
    return checked_sequence

def result_sequence(sequence, max_length):
    sequence1 = generate_text(sequence, max_length)
    sequence2 = sequence1.split(f'{sequence},')[1]
    sequence3 = spell_check(sequence2)
    sequence4 = sequence3.replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ')
    sequence5 =  sequence4.replace('. ', '.. ')
    if len(sequence5.split('. ')) <= 1:
        sequence6 = ' '.join(sequence5.split('. '))
    else:
        sequence6 = ' '.join(sequence5.split('. ')[:-1])
    sequence7 = re.sub('[^ㄱ-ㅎㅏ-ㅣ가-힣0-9,. ]', '', spell_check(sequence6)).replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ')
    return sequence7