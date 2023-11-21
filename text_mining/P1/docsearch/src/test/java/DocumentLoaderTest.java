import org.example.DocumentLoader;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Iterator;

import static org.junit.jupiter.api.Assertions.*;

class DocumentLoaderTest {

    @TempDir
    Path tempDir;

    @Test
    void testLoadNonRecursive() throws IOException {
        // Set up test environment
        Path txtFile = tempDir.resolve("test.txt");
        Files.createFile(txtFile);

        Path subDir = tempDir.resolve("sub");
        Files.createDirectory(subDir);
        Path subTxtFile = subDir.resolve("test.txt");
        Files.createFile(subTxtFile);

        // Test
        DocumentLoader loader = new DocumentLoader(tempDir.toString(), false);
        Iterator<File> iterator = loader.iterator();

        assertTrue(iterator.hasNext());
        assertEquals(txtFile.toFile(), iterator.next());
        assertFalse(iterator.hasNext());
    }

    @Test
    void testLoadRecursive() throws IOException {
        // Set up test environment
        Path subDir = tempDir.resolve("sub");
        Files.createDirectory(subDir);
        Path subTxtFile = subDir.resolve("test.txt");
        Files.createFile(subTxtFile);

        // Test
        DocumentLoader loader = new DocumentLoader(tempDir.toString(), true);
        Iterator<File> iterator = loader.iterator();

        assertTrue(iterator.hasNext());
        assertEquals(subTxtFile.toFile(), iterator.next());
        assertFalse(iterator.hasNext());
    }

    @Test
    void testFileTypes() throws IOException {
        // Set up test environment with different file types
        Files.createFile(tempDir.resolve("test.txt"));
        Files.createFile(tempDir.resolve("test.doc"));
        Files.createFile(tempDir.resolve("test.pdf"));
        Files.createFile(tempDir.resolve("test.docx"));
        Files.createFile(tempDir.resolve("test.jpg")); // Non-supported type

        // Test
        DocumentLoader loader = new DocumentLoader(tempDir.toString(), true);
        Iterator<File> iterator = loader.iterator();

        int count = 0;
        while (iterator.hasNext()) {
            iterator.next();
            count++;
        }

        assertEquals(4, count); // Only 4 supported types should be loaded
    }

    @Test
    void testInvalidDirectory() {
        // Test with an invalid directory
        DocumentLoader loader = new DocumentLoader("non_existent_directory", false);
        Exception exception = assertThrows(RuntimeException.class, loader::iterator);

        String expectedMessage = "Error loading documents";
        String actualMessage = exception.getMessage();

        assertTrue(actualMessage.contains(expectedMessage));
    }
}
