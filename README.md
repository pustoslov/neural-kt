# Description
This is just a training project of a Feed Forward neural network written in Kotlin. It contains three classes for architecture with one, two and three hidden layers. As an example of work, the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset was chosen (you can find it in Main.kt).
# Implementation
## For build.gradle
Add it in your root build.gradle at the end of repositories:
```
repositories {
			maven { url 'https://jitpack.io' }
		}
```
Add the dependency:
```
dependencies {
	        implementation 'com.github.pustoslov:neural-kt:1.0.5'
	}
```
## For build.gradle.kts
Add it in your root build.gradle at the end of repositories:
```kts
repositories {
      maven{
        url = uri("https://jitpack.io")
    }
}
```
Add the dependency:
```kts
dependencies { 
          implementation ("com.github.pustoslov:neural-kt:1.0.5") 
    }
```
# How to use
|Class|Description|
|------|------|
|NeuralNet1L| Neural network with one hidden layer|
|NeuralNet2L| Neural network with two hidden layers|
NeuralNet3L| Neural network with three hidden layers|

### Create object:
```kt
val neuralNet = NeuralNet1L(3, 4, 2, 0.1)
```
As parameters it takes count of input nodes, count of  hidden nodes, count of output nodes and a learning rate
### Train
```kt
val inputsList = ListOf(0.1, 0.2, 0.3)
val targetOutputsList = ListOf(0.01, 0.99, 0.01)
neuralNet.train(inputsList, targetOutputsList)
```
`train()` takes `List<Double>` as inputs and as target outputs
### Test
```kt
neuralNet.query(inputsList)
```
query returns outputs `List<Double>`
### Get and Set Weights
After training you can get weights as `List<List<DoubleArray>>` where every `List<DoubleArray>` are weights of a single layer.
```kt
val weights = neuralNet.getWeights()
```
And set up your optional weights
```kt
val weightsInputToHidden = 
            listOf(doubleArrayOf(0.1142, 0.4334, 0.2238),
                    doubleArrayOf(0.8542, 0.3434, 0.6438),
                    doubleArrayOf(0.9242, 0.1234, 0.6938),
                    doubleArrayOf(0.2342, 0.4434, 0.9138))
val weightsHiddenToOutput = 
            listOf(doubleArrayOf(0.1142, 0.4334, 0.2238, 0.6438),
                    doubleArrayOf(0.9242, 0.1234, 0.6938, 0.9138)
    )

val myWeights = listOf(weightsInputToHidden, weightsHiddenToOutput)

neuralNet.setWeights(myWeights)
```
## License
Copyright © 2023 [pustoslov](https://github.com/pustoslov).
This project is licensed under [Apache 2.0](https://github.com/pustoslov/neural-kt/blob/main/LICENSE).
