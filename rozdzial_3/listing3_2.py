import re
openai_result = '''
* Wprowadzenie
    a. Definicja inżynierii danych
    b. Rola inżynierii danych w dzisiejszym biznesie
* Podsumowanie
    a. Przegląd głównych korzyści z inżynierii danych
    b. Postrzeganie roli inżynierii danych w przyszłości.
'''
section_regex = re.compile(r"\* (.+)")
subsection_regex = re.compile(r"\s*([a-z]\..+)")
result_dict = {} 
current_section = None
for line in openai_result.split("\n"): 
    section_match = section_regex.match(line) 
    subsection_match = subsection_regex.match(line)
    if section_match:
        current_section = section_match.group(1) 
        result_dict[current_section] = []
    elif subsection_match and current_section is not None:
        result_dict[current_section].append(subsection_match.group(1))
print(result_dict)

