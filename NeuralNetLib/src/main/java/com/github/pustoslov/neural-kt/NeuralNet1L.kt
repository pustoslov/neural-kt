package com.github.pustoslov.neural-kt

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import kotlin.math.pow

class NeuralNet1L (inputNodes: Int, hiddenNodes: Int,
                   outputNodes : Int, private val learningRate: Double) {

    private var wInputHidden : NDArray<Double, D2>
    private var wHiddenOutput : NDArray<Double, D2>


    init {
        wInputHidden = mk.rand(from = -inputNodes.toDouble().pow(-0.5),
            until = inputNodes.toDouble().pow(-0.5), hiddenNodes, inputNodes)
        wHiddenOutput = mk.rand(from = -hiddenNodes.toDouble().pow(-0.5),
            until = hiddenNodes.toDouble().pow(-0.5), outputNodes, hiddenNodes)
    }

    fun query(inputsList : List<Double>) : List<Double> {
        val inputs = mk.ndarray(inputsList).reshape(1, inputsList.size).transpose()

        val hiddenInputs = mk.linalg.dot(wInputHidden, inputs)
        val hiddenOutputs = activationFun(hiddenInputs)

        val finalInputs = mk.linalg.dot(wHiddenOutput, hiddenOutputs)
        val finalOutputs = activationFun(finalInputs)

        return finalOutputs.toList()
    }

    fun train(inputsList: List<Double>, targetList: List<Double>) {
        val inputs = mk.ndarray(inputsList).reshape(1,inputsList.size).transpose()
        val targets = mk.ndarray(targetList).reshape(1,targetList.size).transpose()

        val hiddenInputs = mk.linalg.dot(wInputHidden, inputs)
        val hiddenOutputs = activationFun(hiddenInputs)

        val finalInputs = mk.linalg.dot(wHiddenOutput, hiddenOutputs)
        val finalOutputs = activationFun(finalInputs)

        val outputErrors = targets - finalOutputs
        val hiddenErrors = mk.linalg.dot(wHiddenOutput.transpose(), outputErrors)

        wHiddenOutput = wHiddenOutput + (learningRate * (mk.linalg.dot((outputErrors * finalOutputs *
                (1.0 - finalOutputs)), hiddenOutputs.transpose())))
        wInputHidden = wInputHidden + (learningRate * (mk.linalg.dot((hiddenErrors * hiddenOutputs *
                (1.0 - hiddenOutputs)), inputs.transpose())))
    }

    fun getWeights() : List<List<DoubleArray>> {
        val firstLayerWeights = mutableListOf<DoubleArray>()
        val secondLayerWeights = mutableListOf<DoubleArray>()
        for(i in 0 until wInputHidden.toListD2().size) {
            firstLayerWeights += wInputHidden.toListD2()[i].toDoubleArray()
        }
        for(i in 0 until wHiddenOutput.toListD2().size) {
            secondLayerWeights += wHiddenOutput.toListD2()[i].toDoubleArray()
        }
        return listOf(firstLayerWeights, secondLayerWeights)
    }

    fun setWeights(weights: List<List<DoubleArray>>) {
        for (i in 0 until weights[0].size) {
            wInputHidden[i] = mk.ndarray(weights[0][i])
        }
        for (i in 0 until weights[1].size) {
            wHiddenOutput[i] = mk.ndarray(weights[1][i])
        }
    }

}


private fun activationFun(layer : NDArray<Double, D2>) : NDArray<Double, D2> {
    val toList = layer.toMutableList()
    for (i in 0 until toList.count()) {
        toList[i] = 1/(1+Math.E.pow(-toList[i]))
    }
    return mk.ndarray(toList).reshape(1, toList.size).transpose()
}
