TODO
===

## Game implementation
- [ ] Understand the rules of the game and how scoring works
- [ ] Create the board representation in memory
- [ ] Given two consecutive boards, detect which move has been made
- [ ] Given a board and a move, compute the score of the round

## Data extraction from images
- [ ] ???

## Extragere board
- am incercat sa izolez board-ul folosind Canny + filtre => fail din cauza noise-ului din background (lemn)
- am incercat sa fac template match pe tot board-ul si apoi sa extrag board-ul interior => nenecesar de complicat,
dominourile de pe margine incurca, tot nu se obtine un board central perfect aliniat
- am creat un template folosind GIMP + crop + perspective transform din OpenCV
- template matching cu SIFT pentru board-ul interiror => rezultate ok-ish, dar se poate mai bine, lent
- template matching cu ORB + bruteforce hamming => rapid, rezultate bune
- grid peste board-ul extras, nu pare sa fie nevoie de aliniere cu template-ul

## Dominouri
- diferenta absoluta de medii de culoare in grayscale intre board gol si board cu dominouri pe el => fail, foarte multe
false positives la detectie
- detectare de cercuri pe toata imaginea folosind hough => pretty good, dar scapa multe cercuri
- template matching pe un singur bullet pe toata imaginea => bulletul are prea putine features ca sa mearga template
matchig-ul bine
- template matching cu un domnio intreg => dominourile au prea putine features
- hough circles per patch cu transformari morfologice => merge, 19_05 e stricata
- dupa un pic de tunning, hough circles scoate corect toate true positives
- findContour are rezultate proaste pentru gasit liniile mediene de pe domino