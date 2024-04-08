# -*- coding: utf-8 -*-
"""Sales Support Model (hsr) Retrieval Augmented Generation (RAG)"""

from models.hybrid_search_retreiver import HybridSearchRetriever
hsr = HybridSearchRetriever()

if __name__ == "__main__":
   
    dropbox_pdf= '/PDF´s'
    hsr.load(dropbox_folder=dropbox_pdf)