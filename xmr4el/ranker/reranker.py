import os
import torch
import logging
import random

import numpy as np
import torch.nn as nn


from transformers import BertTokenizer, BertModel


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class ReRanker(nn.Module):
    
    def __init__(self, model_name='bert-base-uncased', embedding_dim=768):
        """
        [CLS] mention text [SEP] candidate entity text [SEP], Mention / Candidate / Label
        """
        super(ReRanker, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Combine BERT [CLS] output with label embedding
        self.classifier = nn.Sequential(
            nn.linear(768 + embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # Output a match score
        )

    
    def forward(self, input_ids, attention_mask, label_embeddings):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :] # CLS token
        
        combined = torch.cat([cls_output, label_embeddings], dim=1)
        score = self.classifier(combined)
        return score.squeeze(-1) # shape: (batch_size, )
            
    def train_model(self, mention_texts, kb_indices, conc_embeddings, n_neg_mentions=10,
                    batch_size=16, epochs=3, lr=2e-5, random_state=42):

        # Generate positive and negative training pairs
        mentions, label_embeddings, labels = self.input_pairs(
            mention_texts=mention_texts,
            kb_indices=kb_indices,
            conc_embeddings=conc_embeddings,
            n_neg_mentions=n_neg_mentions,
            random_state=random_state
        )

        # Tokenize mention texts
        inputs = self.tokenizer(
            mentions,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        label_embeddings_tensor = torch.tensor(label_embeddings, dtype=torch.float)
        labels_tensor = torch.tensor(labels, dtype=torch.float)

        dataset = TensorDataset(
            inputs['input_ids'],
            inputs['attention_mask'],
            label_embeddings_tensor,
            labels_tensor
        )

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = AdamW(self.model.parameters(), lr=lr)
        loss_fn = BCEWithLogitsLoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                input_ids, attention_mask, label_embs, lbls = batch
                self.model.zero_grad()
                scores = self.model(input_ids, attention_mask, label_embs)
                loss = loss_fn(scores.squeeze(), lbls)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss:.4f}")

    @torch.no_grad()
    def rank(self, mention, candidate_embeddings, max_length=128):
        self.eval()

        inputs = self.tokenizer(
            [mention] * len(candidate_embeddings),
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        label_embeddings = torch.tensor(candidate_embeddings, dtype=torch.float).to(self.device)

        scores = self(input_ids, attention_mask, label_embeddings)
        ranked_indices = torch.argsort(scores, descending=True).cpu().tolist()
        return ranked_indices, scores.cpu().tolist()
    
    @staticmethod
    def input_pairs(kb_indices, conc_embeddings, n_neg_mentions=10, random_state=42):
        """Generates true labels and random false labels"""
        mentions = []
        label_embeddings = []
        labels = []
        
        random.seed(random_state)
        kb_range = len(kb_indices)
        
        for idx in range(kb_range):
            true_mention = kb_indices[idx]
            # Positive Sample
            mentions.append(true_mention)
            label_embeddings.append(conc_embeddings[idx])
            labels.append(1)
            
            for _ in range(n_neg_mentions):
                rand_idx = random.randrange(kb_range)
                if rand_idx == idx:
                    continue
                mentions.append(true_mention)
                label_embeddings.append(conc_embeddings[rand_idx])
                labels.append(0)
                
        return mentions, label_embeddings, labels
                
        
            
        
