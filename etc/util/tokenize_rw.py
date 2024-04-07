import re


def tokenize(text):
    contractions = {
        "'m": " am",
        "'s": " is",
        "'re": " are",
        "'ll": " will",
        "'d": " would",
        "'ve": " have",
        "isn't": "is not",
        "wasn't": " was not",
        "aren't": " are not",
        "weren't": " were not",
        "don't": " do not",
        "didn't": " did not",
        "won't": " will not",
        "shan't": " shall not",
        "can't": " can not",
        "couldn't": " could not",
        "shouldn't": " should not",
        "wouldn't": " would not",
    }
    pattern = re.compile(r"\b(?:" + "|".join(contractions.keys()) + r")\b")
    text = pattern.sub(lambda match: contractions[match.group(0)], text)

    fileters = "|".join(
        map(re.escape, list('!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n\x97\x96“”'))
    )
    text = re.sub("<.*?>", " ", text, flags=re.S)
    pattern = "|".join(map(re.escape, fileters))
    text = re.sub(pattern, " ", text, flags=re.S)

    return [i.strip().lower() for i in text.split()]
