# Tidy a messy tag list

[![Built on QuiddityML](https://img.shields.io/badge/Built%20on-QuiddityML-e8b44a)](https://quiddityml.com)

People type the same tag in different ways, so a raw count treats "NLP" and " nlp " as two different tags. Clean the text and find out how many different tags there are.

## What I found

The tags looked different because of the inconsistent spacing, letter cases used. I used the strip() method to remove any whitespaces before and after each letter, lower() to turn all the tags into lower cases. The replace() method was used to remove spaces after the commas .The sort method was used to arrange the tags in alphabetical order and improve readability.

## What I built

Your code strips and lowercases a line of survey tags, drops the repeats with a set and prints one sorted summary line.

## Skills

- String methods
- Sets
- Sorting
- f-strings

## Run it

```bash
pip install -r requirements.txt
jupyter notebook project.ipynb
```

---

Built step by step on [QuiddityML](https://quiddityml.com), the ML course where you learn by building. Python for ML, Unit 6: Tuples and Sets.
