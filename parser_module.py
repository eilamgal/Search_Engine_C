import calendar
import re
from datetime import datetime
from nltk.corpus import stopwords
from document import Document
from nltk import PorterStemmer


def parse_hashtags(hashtag):
    if hashtag.find(".") > 0:
        hashtag = hashtag[0:hashtag.find(".")]
    list_of_tokens = [hashtag.lower()]
    if hashtag.find("_") > 0:
        list_of_tokens.extend(
            token.lower() for token in re.sub('([_][a-z]+)', r' ', re.sub('([_]+)', r' ', hashtag[1:])).split())
    else:
        list_of_tokens.extend(
            token.lower() for token in re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', hashtag[1:])).split())
    return list_of_tokens


def parse_number(num, suffix):
    num = float(num)
    if suffix.find('/') > 0:
        return str(num) + " " + suffix, True
    ch = ""
    num, flag = get_suffix(num, suffix)
    if 1000 <= num < 1000000:
        num /= 1000
        ch = "K"
    elif 1000000 <= num < 1000000000:
        num /= 1000000
        ch = "M"
    elif num >= 1000000000:
        num /= 1000000000
        ch = "B"
    if num.is_integer():
        num = int(num)
    else:
        num = round(num, 3)
    return str(num) + ch, flag


def get_suffix(num, suffix):
    flag = False
    suffix = suffix.lower()
    if suffix == "thousand":
        num *= 1000
        flag = True
    elif suffix == "million":
        num *= 1000000
        flag = True
    elif suffix == "billion":
        num *= 1000000000
        flag = True
    return num, flag


CUSTOM_STOPWORDS = ["http", "https", "www", "com", "amp;"]
STOPWORDS_FOR_URL = ['twitter', 'status', 'web', 'co', 'com', 'il', 'net']


class Parse:

    def __init__(self, stemming=False):
        self.stop_words = stopwords.words('english')
        self.stop_words.extend(CUSTOM_STOPWORDS)
        self.porter = PorterStemmer()
        self.stem = stemming

    def tokenize_url(self, url):
        """
        Gets a url and extracts informational tokens and tweet number as referral ids (retweets of others)
        :param url: a url string
        :return: url tokens, referral ids (if there were any)
        """
        if url == "{}" or url is None:
            return None, "0"
        url_split = url.find("\":\"")
        if url_split == -1:
            return
        long_url = url[url_split + 3:]
        long_url = long_url[:len(long_url) - 2]
        tokenized_url = re.sub(r'([#!$%^&?*()={}~`\[\]])|([&=#/\.:\-_]+)', r' \1', long_url).split()
        # extract the the tweet id from the url
        referral_id = "0"
        if len(tokenized_url) >= 2:
            if (tokenized_url[len(tokenized_url) - 2] == "status"
                    and tokenized_url[len(tokenized_url) - 1].isnumeric()):
                referral_id = tokenized_url[len(tokenized_url) - 1]
                del tokenized_url[len(tokenized_url) - 1]

        # return all the tokens from the url without special stop words of url and stop words
        return [w.lower() if (len(w) > 0 and w[0].islower() or w[0] == "#") else w.upper() for w in tokenized_url if
                w.isascii() and w not in self.stop_words and w not in STOPWORDS_FOR_URL], referral_id

    def parse_doc(self, doc_as_list):
        """
        This function takes a tweet document as list and parses it to extract the relevant fields
        :param doc_as_list: a list representing one tweet fields.
        :return: Document object with corresponding fields.
        """
        tweet_id = doc_as_list[0]
        tweet_date = doc_as_list[1]
        full_text = doc_as_list[2]
        url_tokens = self.tokenize_url(doc_as_list[3])[0]
        quote_text = doc_as_list[8]
        # parsing urls
        quote_url_tokens, referral_id1 = self.tokenize_url(doc_as_list[9])
        retweet_url_tokens, referral_id2 = self.tokenize_url(doc_as_list[6])
        referrals = {referral_id1, referral_id2}

        # time parsing
        months = {month: index for index, month in enumerate(calendar.month_abbr) if month}
        splitdate = tweet_date.split(' ')
        hour = int(splitdate[3].split(":")[0])
        minute = int(splitdate[3].split(":")[1])
        second = int(splitdate[3].split(":")[2])
        tweet_timestamp = int(
            datetime(int(splitdate[5]), months[splitdate[1]], int(splitdate[2]), hour, minute, second).timestamp())
        # referrals
        if tweet_id in referrals:
            referrals.remove(tweet_id)
        if referrals == {'0'}:
            referrals = None
        elif '0' in referrals:
            referrals.remove('0')

        term_dict = {}

        tokenized_text, entities = self.parse_text(full_text)
        if quote_text:
            parsed_quote = self.parse_text(quote_text)
            tokenized_text.extend(parsed_quote[0])
            entities.extend(parsed_quote[1])
        if url_tokens:
            tokenized_text.extend(url_tokens)
        if quote_url_tokens:
            tokenized_text.extend(quote_url_tokens)
        if retweet_url_tokens:
            tokenized_text.extend(retweet_url_tokens)

        tweet_length = len(tokenized_text)
        # count the frequency of each term
        for term in tokenized_text:
            if len(term) < 2:
                continue
            if term[len(term) - 1] == ".":
                term = term[0:len(term) - 1]
            if term.lower() not in term_dict.keys() and term.upper() not in term_dict.keys():
                term_dict[term] = 1
            elif term.isupper() and term.lower() in term_dict.keys():
                term_dict[term.lower()] += 1
            elif term.islower() and term.upper() in term_dict.keys():
                term_dict[term] = term_dict[term.upper()] + 1
                del term_dict[term.upper()]
            else:
                term_dict[term] += 1

        entities_dict = {}
        # frequency of entity
        for entity in entities:
            if len(entity) < 2 or not entity.isascii():
                continue
            if entity not in entities_dict.keys():
                entities_dict[entity] = 1
            else:
                entities_dict[entity] += 1

        document = Document(tweet_id=tweet_id, tweet_timestamp=tweet_timestamp, term_doc_dictionary=term_dict,
                            entities_doc_dictionary=entities_dict, referral_ids=referrals, tweet_length=tweet_length)
        return document

    def parse_text(self, text):
        """
        Parse a string and return the relevant tokens and entities found in it
        :param text: text to parse
        :return: tokenized words/ hashtags / number etc, and potential entities that were extracted
        """
        if not text:
            return
        tokens_list = []
        clean_text = ""
        # split = text.split(' ')
        if text[0:2] == "RT":
            text = text[2:]
        entities = [x.group() for x in re.finditer(r'[A-Z]+[a-z]+([\s\-]+[A-Z]+[a-z]+)+', text)]
        split = re.sub(r'(\.)(\.)(\.)*|[!$%^&?*()={}~`]+|\[|\]', r' \1', text).split()

        for i in range(len(split)):
            token = split[i]
            if not token.isascii() or "http" in token.lower() or len(token) == 0:
                continue
            try:
                if token.isalpha():
                    clean_text += token + " "
                # HASHTAGS
                elif token[0] == '#':
                    tokens_list.extend(parse_hashtags(token))

                # TAGS
                elif token[0] == '@':
                    tokens_list.append(token if not token.endswith(':') else token[:len(token) - 1].lower())

                # PERCENTAGES
                elif i < len(split) - 1 and split[i + 1] in ["percent", "percentage"]:
                    number = float(token)
                    if token.isnumeric():  # token is a round number
                        tokens_list.append(token + "%")
                    else:  # token is a float number
                        tokens_list.append(str(number) + "%")
                elif token.endswith('%'):
                    number = float(token[:token.find('%')])
                    tokens_list.append(token)

                # NUMBERS
                elif token.replace(',', '').replace('.', '').isnumeric():
                    number = token.replace(',', '')
                    if i < len(split) - 1:
                        number, next_token_used = parse_number(number, split[i + 1])
                        tokens_list.append(number)
                        if next_token_used:
                            i += 1
                    else:
                        number, next_token_used = parse_number(number, "")
                        tokens_list.append(number)

                # ALL THE REST - REGULAR TOKENS
                else:
                    clean_text += token + " "

            except:
                clean_text += token + " "

        tokens_list.extend(clean_text.split(' '))

        text_tokens_without_stopwords = []

        if self.stem:
            text_tokens_without_stopwords = [self.porter.stem(w).lower() if w[0].islower() or w[0] == "#"
                                             else self.porter.stem(w).upper()
                                             for w in tokens_list if len(w) > 0 and w not in self.stop_words]
        else:
            text_tokens_without_stopwords = [w.lower() if w[0].islower() or w[0] == "#"
                                             else w.upper()
                                             for w in tokens_list if len(w) > 0 and w not in self.stop_words]

        return text_tokens_without_stopwords, entities
