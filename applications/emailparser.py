import imaplib, concurrent.futures, pytz
from fast_mail_parser import parse_email
from dateutil import parser
from email.header import decode_header, make_header
from bs4 import BeautifulSoup
import re

class EmailParser:
    
    def __init__(self, username, password, imap_url, last_refresh, BATCH_SIZE=100):
        self.username = username
        self.password = password
        self.imap_url = imap_url
        self.last_refresh = last_refresh
        self.BATCH_SIZE = BATCH_SIZE

    def isMoreRecentDate(self, date):
        utc = pytz.UTC
        date = parser.parse(date)
        date = date.replace(tzinfo=utc)
        last_refresh = parser.parse(self.last_refresh)
        last_refresh = last_refresh.replace(tzinfo=utc)
        return date >= last_refresh
    
    def getEmailBatch(self, args):
        low, high = args
        connection = imaplib.IMAP4_SSL(self.imap_url)
        connection.login(self.username, self.password)
        connection.select("INBOX")
        result, data = connection.fetch(f"{low}:{high}", '(RFC822)')
        emails = []
        if result == 'OK':
            for i in range(0, len(data), 2):
                try:
                    emaildata = parse_email(data[i][1])
                except:
                    continue
                if not self.isMoreRecentDate(emaildata.date):
                    continue
                email = {
                    "subject": emaildata.subject,
                    "from": "",
                    "body": ""
                }
                for key, value in emaildata.headers.items():
                    if key.lower() == "from":
                        email["from"] = value
                        break
                if len(emaildata.text_plain) == 0 and len(emaildata.text_html) == 0:
                    email["body"] = ""
                elif len(emaildata.text_plain) == 0:
                    email["body"] = emaildata.text_html[0]
                else:
                    email["body"] = emaildata.text_plain[0]
                email["body"] = self.cleanText(email["body"])
                emails.append(email)
        return emails

    def getEmails(self):
        mail = imaplib.IMAP4_SSL(self.imap_url)
        mail.login(self.username, self.password)
        _, messages = mail.select('INBOX')
        num_messages = int(messages[0])
        mail.close()
        i = 1
        batches = []
        while i < num_messages:
            batches.append((i, i + self.BATCH_SIZE - 1))
            i += self.BATCH_SIZE
        rtn = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            emails = executor.map(self.getEmailBatch, batches)
            for email in emails:
                rtn += email
        return rtn
    

    def getTextFromHTML(self, html):
        soup = BeautifulSoup(html, features="html.parser")
        # rip out all scripts and style elements
        for script in soup(["script", "style"]):
            script.extract()    
        text = soup.get_text()
        # break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # drop blank lines
        text = " ".join(chunk for chunk in chunks if chunk)
        return text

    def cleanText(self, text):
        text = str(make_header(decode_header(text)))
        # remove whitespace
        text = " ".join(text.split())
        # remove links
        text = re.sub(r'http\S+', '', text)
        # convert HTML to plain text
        text = self.getTextFromHTML(text)
        # remove characters to prevent cell overflow
        if len(text) > 32000:
            return text[:32000]
        return text