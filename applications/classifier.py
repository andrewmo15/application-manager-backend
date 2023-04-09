import re
from functools import partial
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, logging
from .nlp_models.email_classifier import EmailDataset, transformer_collate_fn
from transformers import AutoTokenizer
logging.set_verbosity_error()

class EmailData:

    def __init__(self, emails):
        self.emails = emails
        self.classifier = self.load_checkpoint('applications/nlp_models/classifier_checkpoint.pth')
        self.ner = self.load_checkpoint('applications/nlp_models/ner_checkpoint.pth')
        self.classifier_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.ner_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def getClassifications(self):
        rtn = []
        for email in self.emails:
            predict = self.getClassification(self.cleanText(email['body']), self.classifier)
            if not predict == "Irrelevant":
                company, position = self.getCompanyAndPosition(self.ner, email['body'])
                rtn.append({"status": predict, "company": company[:100], "position": position[:100]})
        return rtn

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
        for parameter in model.parameters():
            parameter.requires_grad = False

        model.eval()
        return model

    def getClassification(self, email, model):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        df = pd.DataFrame([[email, -1]], columns=['email', 'status'])
        email_dataset = EmailDataset(df)
        email_dataloader = DataLoader(email_dataset,batch_size=1,collate_fn=partial(transformer_collate_fn, tokenizer=self.classifier_tokenizer))
        prediction = 0
        with torch.no_grad():
            for sentences, _, masks in email_dataloader:
                output = model(sentences.to(device), masks.to(device))
                output = F.softmax(output, dim=1)
                prediction = torch.argmax(output, dim=1)
        if prediction == 0:
            return 'Submitted'
        elif prediction == 1:
            return 'Rejected'
        else:
            return 'Irrelevant'

    def cleanText(self, text):
        return re.sub(r'[^A-Za-z0-9 ]+', '', text)
    
    def getCompanyAndPosition(self, model, paragraph):
        label_list = ['O','ORG','POS']
        tokens = self.ner_tokenizer(paragraph)
        inputs = torch.tensor(tokens['input_ids']).unsqueeze(0)
        masks = torch.tensor(tokens['attention_mask']).unsqueeze(0)
        predictions = model.forward(input_ids=inputs, attention_mask=masks)
        predictions = torch.argmax(predictions.logits.squeeze(), axis=1)
        predictions = [label_list[i] for i in predictions]
        words = self.ner_tokenizer.batch_decode(tokens['input_ids'])
        company = self.getValue(predictions, words, "ORG")
        position = self.getValue(predictions, words, "POS")
        return company.title(), position.title()

    def getValue(self, predictions, words, type):
        values = {}
        temp = ""
        flag = False
        for i, word in enumerate(words):
            if flag and predictions[i] == type:
                if "##" in word:
                    temp += word[2:]
                else:
                    temp += " " + word
            elif predictions[i] == type:
                flag = True
                temp += word
            elif flag and not predictions[i] == type:
                try:
                    values[temp] += 1
                except KeyError:
                    values[temp] = 1
                temp = ""
                flag = False
        temp = 0
        rtn = ""
        for key, count in values.items():
            if count > temp:
                rtn = key
        return rtn