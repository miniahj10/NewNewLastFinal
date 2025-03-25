import os


def extract_unames():
    lst = []
    for file in os.listdir('/Users/jaiharishsatheshkumar/PycharmProjects/lastPhase1/user_voice_storage'):
        lst.append(file.split('_')[0])

    return lst


def check_existing(web_input_name):
    existing = extract_unames()
    if web_input_name in existing:
        return False
    return True
