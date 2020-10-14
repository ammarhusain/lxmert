from pathlib import Path
import json

GQA_ROOT = '../questions/'

path = Path(GQA_ROOT)
split2name = {
    'train': 'train',
    'valid': 'val',
    'testdev': 'testdev',
    'test': 'test',
    'challenge': 'challenge'
    }

for split, name in split2name.items():
    with open(path / ("%s_balanced_questions.json" % name)) as f:
        data = json.load(f)
        new_data = []
        for key, datum in data.items():
            new_datum = {
                'question_id': key,
                'img_id': datum['imageId'],
                'sent': datum['question'],
            }
            try:
              if 'answer' in datum:
                  new_datum['label'] = {datum['answer']: 1.}
              if 'semanticStr' in datum:
                  new_datum['semantic_str'] = datum['semanticStr']
              if 'fullAnswer' in datum:                
                 new_datum['full_answer'] = datum['fullAnswer']   
            except:
              print(f"split {split} .... {datum}")
            new_data.append(new_datum)
        json.dump(new_data, open("../%s.json" % split, 'w'),
                  indent=4, sort_keys=True)

