package org.example;

import org.apache.commons.cli.*;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.index.IndexNotFoundException;
import org.apache.lucene.search.highlight.InvalidTokenOffsetsException;
import org.apache.tika.exception.TikaException;

import java.io.IOException;

public class Main {
    private static final Logger logger = LogManager.getLogger(Main.class);

    public static void main(String[] args) {
        CommandLineParser parser = new DefaultParser();

        Options options = getOptions();

        try {
            CommandLine line = parser.parse(options, args);

            if (line.hasOption("index")) {
                String directory = line.getOptionValue("directory");
                boolean recursive = line.hasOption("recursive");
                indexDocuments(directory, recursive);
            } else if (line.hasOption("search")) {
                int topFragments = Integer.parseInt(line.getOptionValue("topFragments", "3"));
                int topDocuments = Integer.parseInt(line.getOptionValue("topDocuments", "5"));
                int fragmentSize = Integer.parseInt(line.getOptionValue("fragmentSize", "50"));

                String directory = line.getOptionValue("directory");
                String query = line.getOptionValue("query");
                searchIndex(directory, query, topDocuments, topFragments, fragmentSize);
            } else {
                throw new ParseException("Invalid command");
            }
        } catch (ParseException exp) {
            logger.error("Parsing failed.", exp);
        } catch (IndexNotFoundException exp) {
            logger.error("Index not found in target folder. Run indexing first.", exp);
        } catch (Exception exp) {
            logger.error("Failed while running.", exp);
        }
    }

    private static Options getOptions() {
        Options options = new Options();
        options.addOption("index", false, "Index documents in a directory");
        options.addOption("search", false, "Search the index");
        options.addOption("recursive", false, "Use recursive search in indexing");
        options.addOption("directory", true, "Directory path");
        options.addOption("query", true, "Search query");

        options.addOption("topFragments", true, "Number of fragment results per document. (default 3)");
        options.addOption("fragmentSize", true, "How wide a fragment should be in results. (default 50)");
        options.addOption("topDocuments", true, "Number of document results. (default 5)");
        return options;
    }

    private static void indexDocuments(String directory, boolean recursive) throws TikaException, IOException {
        var application = new DocumentSearchApplication();
        logger.info("Indexing documents in " + directory + ". It could take a while...");
        application.indexDocuments(directory, recursive);
        logger.info("Indexing done. Index saved, you can perform the search.");
    }

    private static void searchIndex(String directory, String query, int topDocs, int topFrags, int fragSize) throws IOException, org.apache.lucene.queryparser.classic.ParseException, InvalidTokenOffsetsException {
        var application = new DocumentSearchApplication();
        logger.info("Performing search in " + directory + " for: " + query);
        var results = application.searchIndex(directory, query, topDocs, topFrags, fragSize);

        if (results.isEmpty()) {
            logger.warn("No results found.");
            return;
        }

        for (int i = 0; i < results.size(); i++) {
            logger.info("Result " + (i + 1) + ": " + results.get(i));
        }
    }
}