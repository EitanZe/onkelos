import requests
import csv
from regex import regex
from bs4 import BeautifulSoup

no_vowel_hebrew_re = r'[א-ת,#-@, ,]+' # keeps all hebrew characters and symbols without vowels
book_ch_nums = {1:50, 2:40, 3:27, 4:36, 5:34} # number book mapped to number of chapters in that book
book_names = ['Gn', 'Ex', 'Lv', 'Nm', 'Dt']

HEB_TO_LAT = {
    'א': ')',
    'ב': 'b',
    'ג': 'g',
    'ד': 'd',
    'ה': 'h',
    'ו': 'w',
    'ז': 'z',
    'ח': 'x',
    'ט': 'T',
    'י': 'y',
    'כ': 'k',
    'ך': 'K',
    'ל': 'l',
    'מ': 'm',
    'ם': 'M',
    'נ': 'n',
    'ן': 'N',
    'ס': 's',
    'ע': '(',
    'פ': 'p',
    'ף': 'P',
    'צ': 'c',
    'ץ': 'C',
    'ק': 'q',
    'ר': 'r',
    'ש': '$',
    # 'שׂ': '$', not necessary(?)
    'ת': 't',
}

def replace_with_lat_chars(w: str):
    lat = map(lambda c: HEB_TO_LAT.get(c, c), w)

    return ''.join(list(lat))

def heb_to_lat(st: str):
    words = st.split()
    lat_words = list(map(replace_with_lat_chars, words))

    return ' '.join(lat_words)

# book_verses = []
with open('Targum Onkelos Chapters.csv', 'w', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Verse", "Hebrew", "txt"])
    for book in book_ch_nums:
        # urls for chapters are of this format: https://cal.huc.edu/get_a_chapter.php?file=5100<num book>&sub=<num chapter>&cset=H
                        # e.g. genesis 1: https://cal.huc.edu/get_a_chapter.php?file=51001&sub=01&cset=H
        # ch_links = [] # list of links for each chapter
        # book_verses = []


        # for num in range(1,10):
        #     ch_links.append("https://cal.huc.edu/get_a_chapter.php?file=5100"+str(book)+"&sub=0"+str(num)+"&cset=H")
        # for num in range(10,book_ch_nums.get(book)+1):
        #     ch_links.append("https://cal.huc.edu/get_a_chapter.php?file=5100"+str(book)+"&sub="+str(num)+"&cset=H")



        # for link in ch_links:
        link = f'https://cal.huc.edu/get_a_chapter.php?file=5100{book}&cset=H'
        page = requests.get(link)
        soup = BeautifulSoup(page.text, 'html.parser')
        page_text = soup.find_all(["a","bdo"]) # a tags are links, bdo tags used when not a link
        word_list = [word.getText().strip("\n\t") for word in page_text]
        
        word_list = word_list[3:len(word_list)] # first 3 tags unrelated
        if "(" not in word_list[0]: # for not first chapter
            word_list = word_list[1:len(word_list)]
        if "chapter" in word_list[len(word_list)-1]: # for not the last chapter
            word_list = word_list[0:len(word_list)-1]
        
        # go through the words and only keep the relevant ones, to construct a verse list
        verse_list = []
        verse_num = -1
        for word in word_list:
            word = word.replace('וּ', 'ו')
            if "(" in word:
                # rest = word[word.index(")")+1:len(word)]
                # rest = rest[::-1] # reverses the string
                # word = "(1) "+rest

                word = word[word.index(")")+1:len(word)]
                word = word[::-1] # reverses the string
                pair = [str(book)+':'+word[0:word.index(':')+3].strip(), ""]
                
                # for links in verse names, gets rid of doubles. empty check for first verse
                if not (len(verse_list) != 0 and word.strip() in verse_list[len(verse_list)-1][0]):
                    verse_num += 1
                    verse_list.append(pair)
                continue
            elif "<" in word: # don't include words in angle brackets
                continue
            elif word != '' and word[0] == "/" and word[-1] == "/": # if surrounded by slashes then ignore
                continue
            elif '/' in word:
                # if there are multiple options for words from varying sources, only include the first one
                i = word.index('/')
                word = word[0:i]
            if '#' in word:
                continue
            
            
            verse_list[verse_num][1] = verse_list[verse_num][1]+" "+word
        
        # book_verses.extend(verse_list) # for making full book files
        
        '''
        single file separated by chapter
        '''
        ch_list = []
        v1 = ''.join(regex.findall(no_vowel_hebrew_re, verse_list[0][1]))
        latin_text = heb_to_lat(v1)
        ch_list.append([verse_list[0][0][0:4], v1, latin_text])
        for i in range(1,len(verse_list)-1):
            # verse[1] = verse[1].replace('וּ', 'ו')
            verse_list[i][1] = ''.join(regex.findall(no_vowel_hebrew_re, verse_list[i][1]))
            latin_text = heb_to_lat(verse_list[i][1])
            if verse_list[i][0][2:4] not in ch_list[-1][0][2:4]:
                writer.writerow(ch_list[-1])
                # writer.writerow([verse_list[i][0],verse_list[i][1],latin_text])
                ch_list.append([verse_list[i][0][0:4],verse_list[i][1],latin_text])                
            else:
                ch_list[-1][1] = f'{ch_list[-1][1]} {verse_list[i][1]}'
                ch_list[-1][2] = f'{ch_list[-1][2]} {latin_text}'
        writer.writerow(ch_list[-1])

        '''
        single file separated by book
        '''
        # book_list = []
        # v1 = ''.join(regex.findall(no_vowel_hebrew_re, verse_list[0][1]))
        # latin_text = heb_to_lat(v1)
        # book_list.append([verse_list[0][0][0], v1, latin_text])
        # for i in range(1,len(verse_list)-1):
        #     # verse[1] = verse[1].replace('וּ', 'ו')
        #     verse_list[i][1] = ''.join(regex.findall(no_vowel_hebrew_re, verse_list[i][1]))
        #     latin_text = heb_to_lat(verse_list[i][1])
        #     if verse_list[i][0][0] is not book_list[-1][0][0]:
        #         writer.writerow(book_list[-1])
        #         # writer.writerow([verse_list[i][0],verse_list[i][1],latin_text])
        #         book_list.append([verse_list[i][0][0:4],verse_list[i][1],latin_text])                
        #     else:
        #         book_list[-1][1] = book_list[-1][1]+verse_list[i][1]
        #         book_list[-1][2] = book_list[-1][2]+latin_text
        # writer.writerow(book_list[-1])

        '''
        single file separated by verse
        '''
        # for verse in verse_list:
        #     verse[1] = ''.join(regex.findall(no_vowel_hebrew_re, verse[1]))
        #     latin_text = heb_to_lat(verse[1])
        #     writer.writerow([verse[0],verse[1],latin_text])
        
        # assumes there is a folder named "data", including 5 subfolders (Gn, Ex, Lv, Nm, Dt)
        # ch_num = str(ch_links.index(link)+1) if ch_links.index(link)+1 >= 10 else "0"+str(ch_links.index(link)+1)
        
    '''
        # txt per chapter
        with open('data\\'+book_names[book-1]+'\TgO '+
                  book_names[book-1]+' chapter '+ch_num+'.txt', 'w', encoding="utf-8") as f:
            # specifying the encoding allows for hebrew characters to work
            for verse in verse_list:
                verse = verse.replace('וּ', 'ו') # vuv with middle dot single character,
                                                # so regex doesn't remove vowel, need to replace
                verse = ''.join(regex.findall(no_vowel_hebrew_re, verse)) # join concatenates array results of regex
                f.write(verse+"\n")
        
        # csv per chapter
        with open('csv data\\'+book_names[book-1]+'\TgO '+
                  book_names[book-1]+' chapter '+ch_num+'.csv', 'w', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Verse", "Hebrew", "Latin"])
            for verse in verse_list:
                verse[1] = verse[1].replace('וּ', 'ו')
                verse[1] = ''.join(regex.findall(no_vowel_hebrew_re, verse[1]))
                latin_text = heb_to_lat(verse[1])
                writer.writerow([verse[0],verse[1],latin_text])
        
    
    # full book files
    with open(f'whole books\TgO {book_names[book-1]}.csv', 'w', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Verse", "Hebrew", "Latin"])
        for verse in book_verses:
            verse[1] = verse[1].replace('וּ', 'ו')
            verse[1] = ''.join(regex.findall(no_vowel_hebrew_re, verse[1]))
            latin_text = heb_to_lat(verse[1])
            writer.writerow([verse[0],verse[1],latin_text])
'''
# one file
# with open('Targum Onkelos.csv', 'w', encoding="utf-8") as f:
#     writer = csv.writer(f)
#     writer.writerow(["Verse", "Hebrew", "txt"])
#     for verse in book_verses:
#         verse[1] = verse[1].replace('וּ', 'ו')
#         verse[1] = ''.join(regex.findall(no_vowel_hebrew_re, verse[1]))
#         latin_text = heb_to_lat(verse[1])
#         writer.writerow([verse[0],verse[1],latin_text])