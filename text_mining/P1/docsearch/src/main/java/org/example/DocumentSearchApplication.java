package org.example;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.highlight.*;
import org.apache.lucene.store.FSDirectory;
import org.apache.tika.exception.TikaException;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class DocumentSearchApplication {
    public static final String CONTENTS_FIELD = "contents";
    static final boolean isAnsiSupported = !System.getProperty("os.name").toLowerCase().startsWith("windows");

    private static final Logger logger = LogManager.getLogger(DocumentSearchApplication.class);

    public void indexDocuments(String directory, boolean recursive) throws IOException, TikaException {
        DocumentLoader documentLoader = new DocumentLoader(directory, recursive);
        InvertedIndexBuilder indexBuilder = new InvertedIndexBuilder("DocumentIndex", documentLoader, directory + "/indexDirectory", logger);
        indexBuilder.build();
    }

    public List<String> searchIndex(String directory, String queryString, int topDocsCnt, int topFragsCnt, int fragSize) throws IOException, ParseException, InvalidTokenOffsetsException {
        List<String> searchResults = new ArrayList<>();
        try (DirectoryReader reader = DirectoryReader.open(FSDirectory.open(Paths.get(directory + "/indexDirectory/DocumentIndex")))) {
            IndexSearcher searcher = new IndexSearcher(reader);
            Analyzer analyzer = new CustomRomanianAnalyzer();

            QueryParser parser = new QueryParser(CONTENTS_FIELD, analyzer);
            Query query = parser.parse(queryString);
            QueryScorer scorer = new QueryScorer(query);

            Formatter formatter = isAnsiSupported ?
                    new SimpleHTMLFormatter("\u001B[31m", "\u001B[0m") :
                    new SimpleHTMLFormatter("<B>", "</B>");
            Fragmenter fragmenter = new SimpleFragmenter(fragSize);
            Highlighter highlighter = new Highlighter(formatter, scorer);
            highlighter.setTextFragmenter(fragmenter);

            TopDocs topDocs = searcher.search(query, topDocsCnt);
            for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
                @SuppressWarnings("deprecation") Document doc = searcher.doc(scoreDoc.doc);
                var docName = doc.get("name");

                String text = doc.get(CONTENTS_FIELD);
                @SuppressWarnings("deprecation") TokenStream tokenStream = TokenSources.getAnyTokenStream(
                        searcher.getIndexReader(),
                        scoreDoc.doc,
                        CONTENTS_FIELD,
                        analyzer
                );

                searchResults.addAll(
                        Arrays.stream(
                                highlighter.getBestFragments(tokenStream, text, topFragsCnt)
                        ).map(result -> "(" + docName + ")" + result).toList()
                );
            }
        }
        return searchResults;
    }
}

