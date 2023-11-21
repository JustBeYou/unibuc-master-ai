# Document Search Application

**Table of contents**
1. Overview
2. Building
3. Usage

## Overview

The application is built mostly using Apache Lucene and Apache Tika. It is a simple
CLI utility for indexing & searching text documents stored on the disk. There
are two processing pipelines in the application:

1. The Indexing Pipeline
    * a target directory list walked through and the files are read
      using Apache Tika (see `DocumentLoader`)
    * an inverted index is built using Apache Lucene and saved to disk
      in the indexed directory (see `InvertedIndexBuilder`)
    * each document added to the index is processed by a custom Lucene analyzer
      which performs the following steps (see `CustomRomanianAnalyzer`):
        * replaces all cedilla characters to comma characters (e.g. `ț` to `ţ`)
        * converts everything to lowercase
        * removes Romanian stop words
        * applies a Snowball Stemmer to all remaining words
        * converts all UTF-8 characters to ASCII equivalents (i.e. removing diacritics etc)
        * returns a stream of tokens to be indexed
2. The Searching Pipeline
    * an index is loaded from disk from a target directory
    * a search query is prepared using the same text processing steps
      as in the indexing phase (see `CustomRomanianAnalyzer`)
    * the search query is performed on the index and results are ranked
      first by document, then by fragment, matches being highlighted (see `DocumentSearchApplication`)

The main application starts in the `Main` class. Tests are also available.

## Building

Requirements:
* Java JDK 17
* Maven

Open a command line application for your operating system and navigate to the
directory of the project, then type:
```shell
maven package
```

The application will be built and tested. The output should be available in
`./target` directory.

## Usage

Running the application with no arguments will generate the usage help message.
```shell
$ java -jar target/docsearch-1.0-SNAPSHOT.jar
[main] INFO - Usage: java -jar <path> [options here]
[main] INFO - index -> Index documents in a directory
[main] INFO - search -> Search the index
[main] INFO - recursive -> Use recursive search in indexing
[main] INFO - directory -> Directory path
[main] INFO - query -> Search query
[main] INFO - topFragments -> Number of fragment results per document. (default 3)
[main] INFO - fragmentSize -> How wide a fragment should be in results. (default 50)
[main] INFO - topDocuments -> Number of document results. (default 5)
[main] ERROR- Parsing failed.
```

### Indexing

Before using the search functionality, you MUST index the desired target directory.
In order to do so, prepare a directory containing only text, PDFs and DOC(X) files
and the type following command into your terminal:
```shell
java -jar target/docsearch-1.0-SNAPSHOT.jar -index -directory <path to your directory>
```

Example command and output:
```shell
$ java -jar target/docsearch-1.0-SNAPSHOT.jar -index -directory ./test_docs/
[main] INFO - Indexing documents in ./test_docs/. It could take a while...
[main] INFO - Indexing document dostoievski-crima-si-pedeapsa.pdf
[main] INFO - Indexing document amintiri-din-copilarie.txt
[main] INFO - Indexing document Jules-Verne-20000-de-leghe-sub-mari.docx
[main] INFO - Indexing done. Index saved, you can perform the search.
```

If you want to index recursively, add `-recursive` to the arguments.

### Searching

After indexing a directory, you can use the application to search for words
in the documents. For example, a simple search would look like this:
```shell
$ java -jar target/docsearch-1.0-SNAPSHOT.jar -search -directory <directory> -query <word>
```

Example command and output:
```shell
$ java -jar target/docsearch-1.0-SNAPSHOT.jar -search -directory ./test_docs/ -query mama
[main] INFO - Performing search in ./test_docs/ for: mama
[main] INFO - Result 1: (amintiri-din-copilarie.txt) cap, și tiva la mama acasă, și am început a-i
[main] INFO - Result 2: (amintiri-din-copilarie.txt) cum a ars el inima unei mame, așa să-i ardă inima
[main] INFO - Result 3: (amintiri-din-copilarie.txt), în toate părțile. Iar mama lui bădița Vasile își
[main] INFO - Result 4: (amintiri-din-copilarie.txt) cap, și tiva la mama acasă, și am început a-i
[main] INFO - Result 5: (amintiri-din-copilarie.txt) cum a ars el inima unei mame, așa să-i ardă inima
[main] INFO - Result 6: (amintiri-din-copilarie.txt), în toate părțile. Iar mama lui bădița Vasile își
[main] INFO - Result 7: (dostoievski-crima-si-pedeapsa.pdf) jertfit pentru mama ei 
```

**NOTE**: If you are running an ANSI-compatible terminal (on Linux/OSX) you should
see the matched words in red color.

The number of matched documents, matched fragments per document and fragment
size can be configured as follows:

```shell
$ java -jar target/docsearch-1.0-SNAPSHOT.jar -search -directory ./test_docs/ -query mama -topDocuments 1 -topFragments 2 -fragmentSize 30
[main] INFO - Performing search in ./test_docs/ for: mama
[main] INFO - Result 1: (amintiri-din-copilarie.txt) tiva la mama acasă, și
[main] INFO - Result 2: (amintiri-din-copilarie.txt) unei mame, așa să-i ardă inima
```

**NOTE**: diacritics are also supported in the command line
```shell
java -jar target/docsearch-1.0-SNAPSHOT.jar -search -directory ./test_docs/ -query acasă
[main] INFO - Performing search in ./test_docs/ for: acasă
[main] INFO - Result 1: (amintiri-din-copilarie.txt); și trec pe lângă casa noastră, și nu intru acasă
[main] INFO - Result 2: (amintiri-din-copilarie.txt) cap, și tiva la mama acasă, și am început a-i
[main] INFO - Result 3: (amintiri-din-copilarie.txt), mama învăța cu mine acasă și citea acum la ceaslov
```