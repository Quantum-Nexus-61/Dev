import spacy

# Load the English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")

# Input text - to summarize
text = """The Earth is the third planet from the Sun in our solar system. It is the only known celestial body to support life. The Earth's atmosphere consists of various gases, including oxygen and nitrogen. It has a diverse climate, with regions that experience different weather patterns. Earth is also home to a wide range of ecosystems, from lush rainforests to arid deserts. It is an incredible and unique planet. Artificial intelligence (AI) is a field of computer science that focuses on creating intelligent machines capable of performing tasks that typically require human intelligence. AI technologies include machine learning, natural language processing, and computer vision. These technologies have applications in various industries, such as healthcare, finance, and transportation. AI has the potential to revolutionize the way we live and work."""
doc = nlp(text)

# Tokenizing the text and generating sentences
words = [token.text for token in doc]
sentences = [sent.text for sent in doc.sents]

# Create a frequency table to keep the score of each word
freqTable = dict()
for word in words:
    word = word.lower()
    if word in freqTable:
        freqTable[word] += 1
    else:
        freqTable[word] = 1

# Create a dictionary to keep the score of each sentence
sentenceValue = dict()
for sentence in sentences:
    for word, freq in freqTable.items():
        if word in sentence.lower():
            if sentence in sentenceValue:
                sentenceValue[sentence] += freq
            else:
                sentenceValue[sentence] = freq

# Calculate the average value of a sentence from the original text
sumValues = sum(sentenceValue.values())
average = int(sumValues / len(sentenceValue))

# Store sentences into the summary
summary = ""
for sentence in sentences:
    if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
        summary += " " + sentence

print(summary)
