% === Criminal code ===
% Translations using http://old.mpublic.ro/ump/multilingua_compendiu.pdf
article(guilty_art_193).
article(guilty_art_194).

% Battery is violent act made by a person to another, resulting in the hospitalization
% of the victim. 

battery(Defendant, HospitalizationDays) :- 
    violence(ViolentAct, Defendant, Victim),
    hospitalization(VictiomsHospitalization, Victim),
    causality(ViolentAct, VictiomsHospitalization),
    duration(VictiomsHospitalization, HospitalizationDays), !.

% art. 193 battery and other violent acts
% If a person commits battery and the hospitalization period is shorter than 90 days,
% the person is gulity due to article 193.
guilty_art_193(Defendant) :-
    battery(Defendant, HospitalizationDays),
    HospitalizationDays =< 90, !.

% art. 194 aggravated battery
% If a person commits battery and the hospitalization period is longer than 90 days,
% the person is gulity due to article 194.
guilty_art_194(Defendant) :-
    battery(Defendant, HospitalizationDays),
    HospitalizationDays > 90, !.

% General definitions
% A person is guilty if there is any article describing their act.
guilty(Defendant, Article) :- article(Article), call(Article, Defendant), !.

% === Example case ===

% Ion assaulted Vasile.
violence(ion_attacks_vasile, ion, vasile).
% Vasile was hospitalized.
hospitalization(vasiles_hospitalization, vasile).
% The hospitalization lasted 45 days.
duration(vasiles_hospitalization, 45).
% The assault caused the hospitalization.
causality(ion_attacks_vasile, vasiles_hospitalization).

% === Ask for conclusions ===    

% Is Ion guilty of anything?
query(guilty(ion, _)). 

% ==========================
:- initialization(main).

main :-
    forall(query(Q), (Q -> writeln(yes:Q) ; writeln(no:Q))),
    halt.

% ==========================

