import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.store.FSDirectory;
import org.apache.tika.exception.TikaException;
import org.example.DocumentLoader;
import org.example.InvertedIndexBuilder;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class InvertedIndexBuilderTest {

    @TempDir
    Path tempDir;

    @Test
    void testIndexBuilding() throws IOException, TikaException {
        // Create a temporary directory for test documents
        Path testDocsDir = tempDir.resolve("testDocs");
        Files.createDirectory(testDocsDir);

        // Create some test documents
        Path doc1 = testDocsDir.resolve("doc1.txt");
        Files.writeString(doc1, "Hello World");
        Path doc2 = testDocsDir.resolve("doc2.txt");
        Files.writeString(doc2, "Apache Lucene Test");

        // Create a DocumentLoader
        DocumentLoader documentLoader = new DocumentLoader(testDocsDir.toString(), false);

        // Create an InvertedIndexBuilder
        String indexName = "testIndex";
        String indexPath = tempDir.resolve("index").toString();
        InvertedIndexBuilder builder = new InvertedIndexBuilder(indexName, documentLoader, indexPath);

        // Build the index
        builder.build();

        // Verify index directory is created
        assertTrue(Files.exists(tempDir.resolve("index").resolve(indexName)));

        // Verify documents are indexed
        try (var indexReader = DirectoryReader.open(FSDirectory.open(tempDir.resolve("index").resolve(indexName)))) {
            assertEquals(2, indexReader.numDocs());
        }
    }
}
