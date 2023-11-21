package org.example;

import org.apache.tika.Tika;
import org.apache.tika.exception.TikaException;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class DocumentLoader implements Iterable<File> {
    private final String directoryPath;
    private final boolean recursive;

    public DocumentLoader(String directoryPath, boolean recursive) {
        this.directoryPath = directoryPath;
        this.recursive = recursive;
    }

    @Override
    public Iterator<File> iterator() {
        try {
            return loadDocuments().iterator();
        } catch (IOException e) {
            throw new RuntimeException("Error loading documents", e);
        }
    }

    private List<File> loadDocuments() throws IOException {
        try (Stream<Path> paths = Files.walk(Paths.get(directoryPath), recursive ? Integer.MAX_VALUE : 1)) {
            return paths
                    .filter(Files::isRegularFile)
                    .filter(path -> path.toString().matches(".*\\.(txt|doc|docx|pdf)$"))
                    .map(Path::toFile)
                    .collect(Collectors.toList());
        }
    }

    public static String extractContentFromFile(File file) throws IOException, TikaException {
        Tika tika = new Tika();
        try (FileInputStream fis = new FileInputStream(file)) {
            return tika.parseToString(fis);
        }
    }
}
