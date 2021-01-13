# SensoryNetwork

Generates spatial sensory data that accurately mimics what a mobile device might produce when being held and used.

This is a modified fork of an [existing network model](https://github.com/nesl/sensegen), where 50+ deprecated components were upgraded to Tensorflow 3.

Gained insight and learned the math behind more advanced network structures and components such as RNN’s, GRU’s, and LSTM, from trying to understand the [ARXIV document](https://arxiv.org/abs/1701.08886).

Training samples include hundreds of manually scraped datasets from an iPhone 6, and data from countless online sources.

</br>

</br>

**Short progress story**: A few weeks in I trained a network model with manually scraped accelerometer data.

These are what the true values look like:
<img src="https://media.discordapp.net/attachments/717459028246266006/745927414198042684/unknown.png?width=1228&height=676"
     alt=""
     style="float: left; margin-right: 10px;" />


After the data that the network generated was graphed out, I was disappointed by this nonsense:
<img src="https://media.discordapp.net/attachments/717459028246266006/745927495584448612/unknown.png?width=1440&height=662"
     alt=""
     style="float: left; margin-right: 10px;" />


It had a sporadic shape with thousands of jumps between negative and positive values because. I made many fixes to try to correct this, and the most prominent (and obvious one) was to normalize the true data.

After these changes, it outputted a glitched and strange, yet beautiful graph:
<img src="https://media.discordapp.net/attachments/717459028246266006/745928071466713118/unknown.png?width=1326&height=675"
     alt=""
     style="float: left; margin-right: 10px;" />


Through the help of online AI communities and sheer determination, after 3 weeks it was ouputting accelerometer values that are indistinguishiably similar to real data, as far as my eyes could tell:
<img src="https://media.discordapp.net/attachments/624971522422997042/746904655883272352/unknown.png?width=1195&height=676"
     alt=""
     style="float: left; margin-right: 10px;" />


</br>

**Gallery**: Here are examples of Gyroscope data files that finished network models have produced.

Gyro X
<img src="https://media.discordapp.net/attachments/624971522422997042/746905995476992082/unknown.png?width=1133&height=675"
     alt=""
     style="float: left; margin-right: 10px;" />

Gyro Y
<img src="https://media.discordapp.net/attachments/624971522422997042/746906096131899433/unknown.png?width=1088&height=675"
     alt=""
     style="float: left; margin-right: 10px;" />

Gyro Z
<img src="https://media.discordapp.net/attachments/624971522422997042/746906275740123167/unknown.png?width=1152&height=675"
     alt=""
     style="float: left; margin-right: 10px;" />
