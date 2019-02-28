import os
import csv
import json
import codecs

# Splits each line of the file into a dictionary of fields
def loadLines(fileName, fields):
    lines = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]
            lines[lineObj['lineID']] = lineObj
    return lines

# Groups fields of lines from `loadLines` into conversations based on *movie_conversations.txt*
def loadConversations(fileName, lines, fields):
    conversations = []
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
            lineIds = eval(convObj["utteranceIDs"])
            # Reassemble lines
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
            conversations.append(convObj)
    return conversations


# Extracts pairs of sentences from conversations
def extractSentencePairs(conversations):
    qa_pairs = []
    for conversation in conversations:
        # Iterate over all the lines of the conversation
        for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i+1]["text"].strip()
            # Filter wrong samples (if one of the lists is empty)
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs

def get_format_movie_lines(corpus):
    # Define path to new file
    datafile = os.path.join(corpus, "formatted_movie_lines.txt")

    delimiter = '\t'
    # Unescape the delimiter
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))

    # Initialize lines dict, conversations list, and field ids
    lines = {}
    conversations = []
    MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
    MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

    # Load lines and process conversations
    print("\nProcessing corpus...")
    lines = loadLines(os.path.join(corpus, "movie_lines.txt"), MOVIE_LINES_FIELDS)
    print("\nLoading conversations...")
    conversations = loadConversations(os.path.join(corpus, "movie_conversations.txt"),
                                      lines, MOVIE_CONVERSATIONS_FIELDS)

    # Write new csv file
    print("\nWriting newly formatted file...")
    with open(datafile, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
        for pair in extractSentencePairs(conversations):
            writer.writerow(pair)

def get_format_lic_data(corpus):
    datafile = os.path.join(corpus, "train_part.txt")
    formatfile = os.path.join(corpus, "formatted_train_part.txt")
    with open(datafile, 'r')  as freader:
        lines = freader.readlines()
    with open(formatfile, 'w') as fwriter:
        for line in lines:
            json_line = json.loads(line)
            conversation = json_line['conversation']
            goal         = json_line['goal']
            knowledge    = json_line['knowledge']
            
            knowledge_keys = set()
            for k in range(len(knowledge)):
                knowledge_keys.add(knowledge[k][0])
            
            goal_ = []
            for j in range(len(goal)):
                goal_.append( ' '.join(goal[j]) )
            
            for i in range(len(conversation)-1):
                sample = []
                sample.append('\t'.join([conversation[i],conversation[i+1]]))
                sample.append('\t'.join(goal_))
            
                knowledge_ = []
                for key in knowledge_keys:
                    if conversation[i].find(key) != -1:
                        for p in range(len(knowledge)):
                            if ' '.join(knowledge[p]).find(key) != -1:
                                knowledge_.append(' '.join(knowledge[p]))
                    break
                sample.append('\t'.join(knowledge_))
                fwriter.write('|'.join(sample)+'\n')

if __name__ == '__main__':
    #corpus_name = "cornell-movie-dialogs-corpus"
    #corpus = os.path.join("../../public_data", corpus_name)
    #get_format_movie_lines(corpus)
    
    corpus_name = "lic2019"
    corpus = os.path.join("../../public_data", corpus_name)
    get_format_lic_data(corpus)
