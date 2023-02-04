
:- op(498, xfy, [or, and]).
:- op(499, xfy, =>).

:- initialization(main).
main :- 
    load_kb('kb_lab.txt'),
    inputs(Inputs),  
    repeat,
    maplist(ask_rating, Inputs), nl,

    findall(rule(R), rule(R), Rules),
    maplist(eval, Rules, Results),
    writeln("Evaluation of each goal predicate:"),
    maplist(writeln, Results),

    output(Goal, MinValue, MaxValue, Precision),
    find_centroid(aggregate_degree, MinValue, MaxValue, Precision, Result),
    write("Estimated output for "), write(Goal), write(": "), writeln(Result),

    retractall(max_satisfaction(_, _)),
    retractall(rating(_, _)), nl,
    fail.

find_centroid(F, Left, Right, Delta, Centroid) :-
    integrate(F, Left, Right, Delta, IntegralSum, MidXs, Areas),
    midsum(Areas, _, _, IntegralSum, Delta, MidPoint),
    length(Areas, AreasLength),
    RevMidPoint is AreasLength - MidPoint - 1,
    nth0(RevMidPoint,MidXs,Centroid).

midsum([], 0, 0, _, _, _) :- !.
midsum([H | T], S, I, KnownTotalSum, Delta, MidPoint) :- 
    midsum(T, SR, J, KnownTotalSum, Delta, MidPoint),
    I is J + 1,
    S is H + SR,
    R is KnownTotalSum / 2 + Delta,
    L is KnownTotalSum / 2 - Delta,
    ((S < R, S > L, MidPoint = I, !); true).

integrate(F, Left, Right, Delta, Result, MidXs, Areas) :- 
    NewRight is Right - Delta,
    range(Left, Delta, NewRight, Xs),
    maplist(add_half(Delta), Xs, MidXs),
    maplist(call(F), MidXs, Ys),
    maplist(multiply(Delta), Ys, Areas),
    sumlist(Areas, Result).

add_half(B, A, C) :- C is A + 1 / 2 * B.
multiply(B, A, C) :- C is B * A.

range(Start, _, Stop, []) :- Start >= Stop, !.
range(Start, Step, Stop, [Start | Other]) :-
    Start =< Stop,
    Next is Start + Step,
    range(Next, Step, Stop, Other).

aggregate_degree(X, Y) :- 
    output(Goal, _, _, _),  
    findall(Predicate, rule(_ => Goal/Predicate), GoalPredicates),
    maplist(capped_degree(Goal, X), GoalPredicates, Ys),
    max_list(Ys, Y).

capped_degree(Input, X, Predicate, Y) :- 
    degree(Input/Predicate, X, Y0),
    max_satisfaction(Input/Predicate, M),
    Y is min(Y0, M).

eval(rule(Antecedents => Consequent), Result) :- 
    eval_consequent(rule(Antecedents => Consequent), ConsequentResult),
    assertz(max_satisfaction(Consequent, ConsequentResult)),
    Result = [Consequent, ConsequentResult].

eval_consequent(rule(A => _), Result) :- eval_antecedent(A, Result).

eval_consequent(rule(A or B => _), Result) :-
    eval_antecedent(A, ResultA),
    eval_antecedent(B, ResultB),
    Result is max(ResultA, ResultB).

eval_consequent(rule(A and B => _), Result) :- 
    eval_antecedent(A, ResultA),
    eval_antecedent(B, ResultB),
    Result is min(ResultA, ResultB).

eval_antecedent(Input/Predicate, Result) :-
    rating(Input, Rating),
    degree(Input/Predicate, Rating, Result).

ask_rating(Input) :-
    write(Input), write(" > "),
    read_line_to_string(user_input, RatingString),
    (RatingString = "stop" -> halt; true),
    number_string(RatingNumber, RatingString),
    assertz(rating(Input, RatingNumber)).

lin_coef(dec, X1, X2, A, B) :- A is 1 / (X1 - X2), B is - X2 / (X1 - X2).
lin_coef(inc, X1, X2, A, B) :- A is 1 / (X2 - X1), B is - X1 / (X2 - X1).

lin(Dir, Left, Right, X, Y) :- 
    Left =< X, X =< Right,
    lin_coef(Dir, Left, Right, A, B),
    Y is A * X + B.

:- dynamic inputs/1.
:- dynamic output/1.
:- dynamic degree/3.
:- dynamic rule/1.
load_kb(Filename) :-
	see(Filename),
    repeat,
        read(Term),
        ( Term == end_of_file; assertz(Term), fail),
	seen.