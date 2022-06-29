# LoRa + RL for frontier nodes  

## Resumen
El principal objetivo es lograr optimizar el gasto de energía en redes LoRa de nodos frontera. Se propone utilizar tales nodos para aumentar las capacidades de la red, redireccionando el tráfico de nodos externos que no están en contacto con el gateway. Esto evita la instalación de nuevos gateways, que en muchos casos no es rentable. Evaluando diferentes configuraciones de transmisión (e.g., Spreading Factor, Coding Rate, Bitrate, etc.), se propone un algoritmo de aprendizaje reforzado para obtener aquellas configuraciones que logran una mayor eficiencia energética, maximizando la vida de la batería del dispositivo, mientras se garantiza una mayor fiabilidad (PRR y PDR), y se premia la transmisión los mensajes de alta prioridad.

## Importante!
**Iterate over main_LoRaRL.docx to report everything.**

Please, install Mendeley plug-in in Word to keep references up.

## Some useful things
Some links:
* https://github.com/LoRaWanRelay/LoRaWanRelay
* https://www.thethingsnetwork.org/forum/t/multiple-lora-node-communication/45490

Imitation learning should be included to speed up training:
* https://stable-baselines3.readthedocs.io/en/master/guide/imitation.html

Smart Battery calculator (to compute energy consumption):
* https://saft4u.saftbatteries.com/en/iot/simulator
