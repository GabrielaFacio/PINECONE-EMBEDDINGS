# -*- coding: utf-8 -*-
"""A class to manage the lifecycle of Pinecone vector database indexes."""

# general purpose imports
import json
import logging
import os
import pyodbc
import dropbox

# pinecone integration
import pinecone
from pinecone import Pinecone, PodSpec
from langchain.schema import BaseMessage, HumanMessage, SystemMessage

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import Document
from langchain.vectorstores.pinecone import Pinecone as LCPinecone

# this project
from models.conf import settings


logging.basicConfig(level=logging.DEBUG if settings.debug_mode else logging.ERROR)


# pylint: disable=too-few-public-methods
class TextSplitter:
    """
    Custom text splitter that adds metadata to the Document object
    which is required by PineconeHybridSearchRetriever.
    """

    def create_documents(self, texts):
        """Create documents"""
        documents = []
        for text in texts:
            # Create a Document object with the text and metadata
            #document = Document(page_content=text, metadata={"context": text})
            document = Document(page_content=text, metadata={"context":text})            
            documents.append(document)
        return documents
    
    
class PineconeIndex:
    """Pinecone helper class."""

    _index: pinecone.Index = None
    _index_name: str = None
    _text_splitter: TextSplitter = None
    _openai_embeddings: OpenAIEmbeddings = None
    _vector_store: LCPinecone = None
    _pinecone_instance:Pinecone=None
 

    def __init__(self, index_name: str = "rag"):
       
        self._pinecone_instance=Pinecone(
            api_key=settings.pinecone_api_key.get_secret_value(),
            host='https://rag-wib1770.svc.gcp-starter.pinecone.io'
            )        
        
        self.index_name = index_name
        logging.debug("PineconeIndex initialized with index_name: %s", self.index_name)
        logging.debug(self.index_stats)
        
           
        self.message_history=[]
    def add_to_history(self,message:BaseMessage):
        self.message_history.append(message)
    def get_history(self):
        return self.message_history

    @property
    def index_name(self) -> str:
        """index name."""
        return self._index_name

    @index_name.setter
    def index_name(self, value: str) -> None:
        """Set index name."""
        if self._index_name != value:
            self.init()
            self._index_name = value
            
    @property
    def index(self) -> pinecone.Index:
        """pinecone.Index lazy read-only property."""
        if self._index is None:
            self.init_index()
            self._index =self._pinecone_instance.Index(index_name=self.index_name, host='https://rag-wib1770.svc.gcp-starter.pinecone.io')
        return self._index

    @property
    def index_stats(self) -> dict:
        """index stats."""
        retval = self.index.describe_index_stats()
        return json.dumps(retval.to_dict(), indent=4)

    @property
    def initialized(self) -> bool:
        """initialized read-only property."""
        pc=Pinecone(api_key=settings.pinecone_api_key.get_secret_value())
        indexes = pc.list_indexes()        
        return self.index_name in indexes.names()

    @property
    def vector_store(self) -> LCPinecone:
        """Pinecone lazy read-only property."""
        if self._vector_store is None:
            if not self.initialized:
                self.init_index()
            self._vector_store = LCPinecone(
                index=self.index,
                embedding=self.openai_embeddings,
                text_key=settings.pinecone_vectorstore_text_key,
            )
        return self._vector_store

    @property
    def openai_embeddings(self) -> OpenAIEmbeddings:
        """OpenAIEmbeddings lazy read-only property."""
        if self._openai_embeddings is None:
            # pylint: disable=no-member
            self._openai_embeddings = OpenAIEmbeddings(
                api_key=settings.openai_api_key.get_secret_value(),
                organization=settings.openai_api_organization,
            )
        return self._openai_embeddings

    @property
    def text_splitter(self) -> TextSplitter:
        """TextSplitter lazy read-only property."""
        if self._text_splitter is None:
            self._text_splitter = TextSplitter()
        return self._text_splitter
    

    def init_index(self):
        """Verify that an index named self.index_name exists in Pinecone. If not, create it."""
        pc=Pinecone(api_key=settings.pinecone_api_key.get_secret_value())
        indexes= pc.list_indexes()
        if self.index_name not in indexes.names():
            print("Indexes does not exist. Creating...")
            pc.create_index(
                name=settings.pinecone_index_name,
                dimension=settings.pinecone_dimensions,
                metric="dot",
                spec=PodSpec(
                    environment="gcp-starter"
                )
            )
            print("Index created.")
        
        else:
            print("Index already exists.")
    def init(self):
        """Initialize Pinecone."""
        # pylint: disable=no-member
        
        self._index = None
        self._index_name = None
        self._text_splitter = None
        self._openai_embeddings = None
        self._vector_store = None

    def listar_elementos(dbx,ruta):
        try:
            #Lista los archivos y carpetas en la carpeta raiz
            response=dbx.files_list_folder(ruta)
            for entry in response.entries:
            #Verifica si entry es un folder
                if isinstance(entry,dropbox.files.FolderMetadata):
                    #Si es folder llama a la función de forma recursiva
                    PineconeIndex.listar_elementos(dbx,entry.path_display)
                elif entry.name.endswith('.pdf'):
                #Si es un archivo PDF, descargarlo y procesarlo
                    print(f'Loading PDF: {entry.name}')
                    _,response=dbx.files_download(entry.path_display)
                    pdf_content=response.content                    
                    loader = PyPDFLoader(pdf_content) 
                    docs = loader.load()
                    
                    for doc in docs:                
                        documents =PineconeIndex.text_splitter.create_documents(texts=[doc.page_content])
                        document_texts = [doc.page_content for doc in documents]
                        embeddings = PineconeIndex.openai_embeddings.embed_documents(document_texts)
                        documents_batch=[]
                        #Upsert documents and embeddings into the existing index
                        for document, embedding in zip(documents, embeddings):
                            documents_batch.append((document.metadata["context"],embedding))
                            if len(documents_batch) >=100:
                                PineconeIndex.index.upsert(documents_batch,namespace="espacio2")
                                documents_batch.clear()

        except Exception as e:
            print(f'Error: {e}')

    
    def pdf_loader(self, dropbox_folder: str):
        """
        Embed PDF (Recursive).
        1. Load PDF document text data
        2. Split into pages
        3. Embed each page
        4. Store in Pinecone using upsert

        Note: it's important to make sure that the "context" field that holds the document text
        in the metadata is not indexed. Currently you need to specify explicitly the fields you
        do want to index. For more information checkout
        https://docs.pinecone.io/docs/manage-indexes#selective-metadata-indexing
        """
        if not self.initialized:
            print("Index is not initialized. Please initialize it first.")
            self.init_index()
        
        # Dropox connection
        access_token = ''
        dbx = dropbox.Dropbox(access_token)

        #Call the function to start the recursive upload
        PineconeIndex.listar_elementos(dbx,dropbox_folder)

    def tokenize(self,text):
        if text is not None:
            return text.split()
        else:
            return[]
    
    def load_sql(self,sql):
        """
        Load data from SQL database
        """
        self.initialize()
        
        #Establecer conexión a la base de datos
        connectionString =("DRIVER={};""SERVER=;" "DATABASE=;""UID=;""PWD=;""TrustServerCertificate=yes;")
        
        conn=pyodbc.connect(connectionString)
        cursor=conn.cursor()

        #ejecutar consulta SQL
        sql="SELECT clave, nombre, certificacion, disponible, tipo_curso_id, sesiones, pecio_lista, tecnologia_id, subcontratado, pre_requisitos, complejidad_id FROM cursos_habilitados WHERE disponible = 1 OR subcontratado = 1"
        cursor.execute(sql)
        rows=cursor.fetchall()
        
        #Procesar cada fila y crear documentos
        for row in rows:
            content=" ".join(str(col) for col in row if col is not None)
            tokens=self.tokenize(content)
            document=Document(
                page_content=content,
                metadata={
                    "context": content,
                    "tokens":tokens
                 })
            
        #Embed the document
            embeddings=self.openai_embeddings.embed_documents([content])
            self.vector_store.add_documents(documents=[document], embeddings=embeddings)
        print("Finished loading data from SQL. \n"+ self.index_stats)
        conn.close()

        
