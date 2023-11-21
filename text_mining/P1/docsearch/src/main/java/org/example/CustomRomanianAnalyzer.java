package org.example;

import org.apache.lucene.analysis.*;
import org.apache.lucene.analysis.ro.RomanianAnalyzer;
import org.apache.lucene.analysis.snowball.SnowballFilter;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.miscellaneous.ASCIIFoldingFilter;

import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.tartarus.snowball.ext.RomanianStemmer;

import java.io.IOException;

public class CustomRomanianAnalyzer extends Analyzer {

    private final CharArraySet stopWords;

    public CustomRomanianAnalyzer() {
        this.stopWords = RomanianAnalyzer.getDefaultStopSet();
    }

    @Override
    protected TokenStreamComponents createComponents(String fieldName) {
        Tokenizer source = new StandardTokenizer();
        TokenStream filter = new CustomCharFilter(source);
        filter = new LowerCaseFilter(filter);
        filter = new StopFilter(filter, stopWords);
        filter = new SnowballFilter(filter, new RomanianStemmer());
        filter = new ASCIIFoldingFilter(filter);
        return new TokenStreamComponents(source, filter);
    }

    private static class CustomCharFilter extends TokenFilter {
        private final CharTermAttribute charTermAttr = addAttribute(CharTermAttribute.class);

        public CustomCharFilter(TokenStream input) {
            super(input);
        }

        @Override
        public final boolean incrementToken() throws IOException {
            if (input.incrementToken()) {
                char[] buffer = charTermAttr.buffer();
                int length = charTermAttr.length();
                for (int i = 0; i < length; i++) {
                    if (buffer[i] == 'ş') buffer[i] = 'ș';
                    else if (buffer[i] == 'ţ') buffer[i] = 'ț';
                    else if (buffer[i] == 'Ţ') buffer[i] = 'Ț';
                    else if (buffer[i] == 'Ş') buffer[i] = 'Ș';
                    else if (buffer[i] == '\n') buffer[i] = ' ';
                }
                return true;
            }
            return false;
        }
    }
}
