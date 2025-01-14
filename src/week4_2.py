from openai import OpenAI
import pandas as pd
import json
import os

openai_key = os.getenv("OPENAI_API_KEY")

def prediction(user_prompt):
  client = OpenAI(openai_key)

  temperature = 0.5
  max_tokens = 4096
  frequency_penalty = 0.0

  # [MYCODE] asistant_prompt로 힌트 제공.
  assistant_prompt = "당신은 한국어에 능통한 사람 입니다. 가장 정답이 높은 번호를 선택해주세요. 답변은 오직 정답인 숫자만 해주세요. 정답은 오직 하나 입니다."

  message=[{"role": "assistant", "content": assistant_prompt}, {"role": "user", "content": user_prompt}]
  response = client.chat.completions.create(
      model="gpt-4o",
      messages=message,
      n=1,
      max_tokens=max_tokens,
      temperature=temperature,
      frequency_penalty=frequency_penalty
  )
  answer = response.choices[0].message.content
  print(f'gtp가 추론한 정답: {answer}')

  return answer

# [MYCODE] 정답이 틀릴 경우 틀린 이유를 분석해서 다시 선택하도록 유도.
def predict_again(question, uncorrect_answer):
  client = OpenAI(openai_key)

  temperature = 0.5
  max_tokens = 4096
  frequency_penalty = 0.0

  # [MYCODE] assistent로 어떤 문제인지 알려줌.
  assistant_prompt = f"문제는 다음과 같습니다. {question}"

  # [MYCODE] user로 오답이 무엇인지 알려주고 틀린 이유를 분석 하도록 한 이후에 다시 정답을 선택하도록 함.
  user_prompt = f"당신은 오답 {uncorrect_answer}번을 선택하였습니다. 틀린 이유를 분석하고 정답을 다시 선택해주세요. 대답은 숫자로만 해주세요."
  message=[{"role": "assistant", "content": assistant_prompt},{"role": "user", "content": user_prompt}]
  response = client.chat.completions.create(
      model="gpt-4o",
      messages=message,
      n=1,
      max_tokens=max_tokens,
      temperature=temperature,
      frequency_penalty=frequency_penalty
  )
  answer = response.choices[0].message.content
  print(f'오답에 따른 재 정답: {answer}')

  return answer

file_path = './2023_11_KICE.json'

with open(file_path, 'r', encoding='utf-8') as f:
  json_data = json.load(f)

total_score = 0
for data in json_data:
  prompt = f"다음 텍스트를 읽고, 질문에 대한 정답을 선택해 주세요:\n\n{data['paragraph']}\n\n"

  problems = data['problems']
  for problem in problems:
    prompt = f"질문: {problem['question']}\n"

    if 'question_plus' in problem:
      prompt += f"질문에 대한 추가적인 정보 입니다.\n{problem['question_plus']}\n\n"

    prompt += "이제 아래부터 정답을 고를 선택사항들 입니다.\n"
    for idx, choice in enumerate(problem['choices']):
      prompt += f"{idx + 1}. {choice}\n"

    prompt += "\n정답은 무엇인가요? (정답 번호를 선택해 주세요)\n\n"

    # [MYCODE] 정답을 선택할 수 있는 기회를 3번을 준다.
    for idx in range(0, 3):
      predict_answer = prediction(prompt)
      predict_answer = predict_answer[:1]
      predict_answer = int(predict_answer)
      answer = problem['answer']

      if answer == predict_answer:
        score = problem['score']
        total_score += score
        break
      else:
        predict_answer = predict_again(prompt, predict_answer)
        predict_answer = predict_answer[:1]
        predict_answer = int(predict_answer)
        if answer == predict_answer:
          score = problem['score']
          total_score += score
          break

print(f'총 점수는 {total_score} 입니다.')
