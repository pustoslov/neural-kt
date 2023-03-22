import com.github.pustoslov.neuralnetlib.NeuralNet1L
import com.github.doyaaaaaken.kotlincsv.dsl.csvReader

// Example of training neural network with one hidden layer on MNIST dataset.

fun main(args: Array<String>) {

    val neuralNet = NeuralNet1L(784, 100,
        10, 0.1)
    var errorsCount = 0

    // Here training dataset goes through training
    // process five times.
    for (i in 0 until 5) {
        csvReader().open("src/main/resources/train.csv") {
            readAllAsSequence().forEach { row: List<String> ->

                val inputs = mutableListOf<Double>()
                val unprepInputs = row.drop(1)
                for (i in 0 until unprepInputs.size) {
                    inputs += ((unprepInputs[i].toDouble() / 255) * 0.99) + 0.1
                }

                val targetNum = row[0].toInt()
                val targetOutput = mutableListOf<Double>(
                    0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                    0.01, 0.01
                )
                targetOutput[targetNum] = 0.99

                neuralNet.train(inputs, targetOutput)
            }
        }
    }

    // Testing the results of training
    csvReader().open("src/main/resources/test.csv"){
        readAllAsSequence().forEach { row: List<String> ->

            val unprepInputs = row.drop(1)
            val inputs = mutableListOf<Double>()
            for (i in 0 until unprepInputs.size) {
                inputs += ((unprepInputs[i].toDouble()/255) * 0.99) + 0.1
            }

            val targetNum = row[0].toInt()

            val output = neuralNet.query(inputs).toDoubleArray()
            val result = output.indices.maxBy { output[it] }

            // Counting errors
            if (result != targetNum) errorsCount++
        }
    }

    // And displaying percentage of accuracy
    val percentage = ((10000-errorsCount).toDouble()/10000.0) * 100
    println("$percentage %")
}
