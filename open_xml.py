from lxml import etree
from nltk.corpus import stopwords
import nltk
import gensim
import re
from collections import defaultdict
import string


class SentenceGenerator(object):
    def __init__(self, docs):
        self.documents = docs  # list
        self.punc = r"""!"#$%&'()*+,-./:;<=>?[\]^_`{|}~"""
        self.trans = str.maketrans('', '', self.punc)
        self.doc_nums = len(self.documents)

    def __iter__(self):
        for doc in self.documents:
            print('Processing document {}'.format(self.doc_nums))
            t = ' '.join(doc.splitlines())
            sents = nltk.sent_tokenize(t)
            self.doc_nums -= 1
            i = 0
            for sent in sents:  # type: str
                i += 1
                print('Processing sentence {}'.format(i))
                sent = sent.translate(self.trans).split()
                filt = [word for word in sent if word not in stopwords.words()]
                yield filt

if __name__ == "__main__":
    documents = []
    print('Extracting body texts')
    frequency = defaultdict(int)
    sentences = []
    for event, element in etree.iterparse(
            'E:\\RESOURCES\TEST_RESOURCES\\tomes\\data\\mboxes\\OfficeoftheGovernor_UO\\eaxs_xml\\GovPerdue.xml',
            tag='BodyContent', remove_blank_text=True, huge_tree=True):
        for child in element:
            if child.tag == 'Content':
                try:
                    if re.search("Content-Transfer-Encoding: base64", child.text):
                        continue
                except TypeError as e:
                    continue
                documents.append(child.text)
        element.clear()
    fh = open('finalizing.txt', 'w')
    print('Setting WordVecs')
    print('{} documents to process'.format(len(documents)))
    sentences = SentenceGenerator(documents)
    model = gensim.models.Word2Vec(sentences, min_count=10, size=300, workers=4)
    fh.write('Finished Extracting body texts')
    model.save('govperdue.vecm')
    fh.close()









