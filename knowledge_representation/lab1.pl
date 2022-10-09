% Lab 1

% introduction
parent(ion,maria).
parent(ana,maria).
parent(ana,dan).
parent(maria,elena).
parent(maria,radu).
parent(elena,nicu).
parent(radu,george).
parent(radu,dragos).

brother(X, Y) :- parent(Z, X), parent(Z, Y).
grandparent(X, Y) :- parent(X, Z), parent(Z, Y).

% exercices

% 1.
max(X, Y, X) :- Y =< X, !.
max(X, Y, Y) :- X =< Y, !.

% 2. 
member(X, [X | _]) :- !.
member(X, [_ | T]) :- member(X, T).

append([], X, [X]) :- !.
append([H | T], X, [H | W]) :- append(T, X, W).

concat(L1, [], L1) :- !.
concat(L1, [H | T], W2) :- 
    append(L1, H, W1),
    concat(W1, T, W2).

% 3.
alternateSum([], 0) :- !.
alternateSum([X], X) :- !.
alternateSum([X, Y | T], S) :-
    alternateSum(T, S1),
    S is X - Y + S1, !.

% 4.

removeFirst([], _, []) :- !.
removeFirst([H | T], H, T) :- !.
removeFirst([H | T], X, [H | W]) :- removeFirst(T, X, W).

removeAll([], _, []) :- !.
removeAll([H | T], H, W) :- removeAll(T, H, W), !.
removeAll([H | T], X, [H | W]) :- removeAll(T, X, W).
 
% 5. 
reverse([], []) :- !.
reverse([X], [X]) :- !.
reverse([H | T], W) :- reverse(T, W1), append(W1, H, W).

remove([H | T], H, T).
remove([H | T], X, [H | W]) :- remove(T, X, W).

perm([], []).
perm([H | T], P) :- perm(T, W), remove(P, H, W).

% 6.

count([], _, 0).
count([H | T], H, W) :- count(T, H, W1), W is W1 + 1, !.
count([_ | T], X, W) :- count(T, X, W), !.

% 7.

take(_, N, W) :- N =< 0, !, N =:= 0, W = [].
take(_, [], []).
take([H | T], N, [H | W]) :- M is N - 1, take(T, M, W).

split(L, I, A, B) :- take(L, I, A), concat(A, B, L).

insert(L, I, X, W) :- 
    J is I - 1, 
    split(L, J, A, B), 
    append(A, X, A1),
    concat(A1, B, W).

% 8.

mergeSorted([], [], []) :- !.
mergeSorted(A, [], A) :- !.
mergeSorted([], B, B) :- !.
mergeSorted([H1 | T1], [H2 | T2], W) :-
    H1 =< H2,
    !,
    mergeSorted(T1, T2, K),
    concat([H1, H2], K, W).
mergeSorted([H1 | T1], [H2 | T2], W) :-
    H2 < H1,
    !,
    mergeSorted(T1, T2, K),
    concat([H2, H1], K, W).
