package org.example;

import org.apache.logging.log4j.Logger;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.tika.exception.TikaException;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;

public class InvertedIndexBuilder {

    private final DocumentLoader documentLoader;
    private final String indexDirectoryPath;
    private final String indexName;
    private final Logger logger;

    public InvertedIndexBuilder(String indexName, DocumentLoader documentLoader, String indexDirectoryPath, Logger logger) {
        this.indexName = indexName;
        this.documentLoader = documentLoader;
        this.indexDirectoryPath = indexDirectoryPath;
        this.logger = logger;
    }

    public void build() throws IOException, TikaException {
        Directory indexDirectory = FSDirectory.open(Paths.get(indexDirectoryPath, indexName));
        IndexWriterConfig config = new IndexWriterConfig(new CustomRomanianAnalyzer());
        try (IndexWriter indexWriter = new IndexWriter(indexDirectory, config)) {
            for (File file : documentLoader) {
                logger.info("Indexing document " + file.getName());
                indexDocument(file, indexWriter);
            }
        }
    }

    private void indexDocument(File file, IndexWriter indexWriter) throws IOException, TikaException {
        Document document = new Document();
        String content = DocumentLoader.extractContentFromFile(file);
        document.add(new TextField("name", file.getName(), Field.Store.YES));
        document.add(new TextField("contents", content, Field.Store.YES));
        indexWriter.addDocument(document);
    }
}
