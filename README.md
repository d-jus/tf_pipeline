# CEL
Celem zadania jest porównanie wyników uzyskanych w roku 2018 w środowisku Matlab  względem uzyskanych w środowisku TF po treningu zgodnie z obecnym stanem wiedzy. Jeżeli wynik **MSE** będzie porównywalny lub nie zostaną stwierdzone mankamenty modelu podstawowego `(2018)`, wykonany zostanie również etap polegający na uruchomieniu SSN w środowisku `Docker` z obsługą zapytań `REST API`. Przygotowanie do głównego zadania poprzedzono wykonaniem dwóch etapów: <br><br>
**etap 1)** transfer `sieć opracowanej w środowisku Matlab` do języka Python  (postać analityczna)<br><br>
**etap 2)** transfer współczynników W i b (wag i obciążeń) `do środowiska Keras`<br><br>
Każdorazowo wyniki predykcji były weryfikowane wzglądem pierwotnych wartości (sieć z 2018). Wykorzystanie `TF` ma na celu użycie technik opracowanych dla potrzeb głębokiego uczenia. Kryterium wyboru nowej sieci będzie `MSE_new > MSE_old` (przy założeniu prawidłowo uzyskanego modelu tj. braku przetrenowania itp.). <br><br>

## UŻYCIE
Obecnie refaktoryzuję kod pozwalający stworzyć potok w oparciu wyłącznie o tf. Wewnętrze problemy Keras z warstwą tf.keras.layers.Normalization opóźniły etap utworzenia sieci w kontenerze z obsługą Rest API.
