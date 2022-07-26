# CEL
Celem zadania jest porównanie wyników uzyskanych w roku 2018 w środowisku Matlab  wzlędem uzyskanych w środowisku TF po treningu zgodnie z obecnym stanem wiedzy. Jeżeli wynik **MSE** będzie porównywalny lub nie zostaną stwierdzone mankamenty modelu podstawowego `(2018)`, wykonany zostanie również krok **7, 8**. Krok **9** polega na uruchomieniu SSN z użyciem środowiska `Docker` i zapytania `REST API`. Przygotowanie do głównego zadania poprzedzono wykonaniem dwóch etapów: <br><br>
**etap 1)** transfer `sieć opracowanej w środowisku Matlab` do języka python  (postać analityczna)<br><br>
**etap 2)** transfer współczyników W i b (wag i obciążeń) `do środowiska Keras`<br><br>
Każdorazowo wyniki predykcji były weryfikowane wglądem pierwotnych wartości (sieć z 2018). Wykorzystanie `TF` ma na celu użycie technik opracowanych dla potrzeb głębokiego uczenia. Kryterium wyboru nowej sieci będzie `MSE_new > MSE_old` (przy założeniu prawidłowo uzyskanego modelu tj. braku przetrenowania itp.). <br><br>

## UŻYCIE