package com.example.neuralnetlib

import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.rand
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import kotlin.math.pow

class NeuralNet2L (inputNodes: Int, hiddenNodes1: Int, hiddenNodes2: Int,
                   outputNodes : Int, private val learningRate: Double) {

    private var wInputHidden : NDArray<Double, D2>
    private var wHiddenHidden : NDArray<Double, D2>
    private var wHiddenOutput : NDArray<Double, D2>



    init {
        wInputHidden = mk.rand(from = -inputNodes.toDouble().pow(-0.5),
            until = inputNodes.toDouble().pow(-0.5), hiddenNodes1, inputNodes)
        wHiddenHidden = mk.rand(from = -hiddenNodes1.toDouble().pow(-0.5),
            until = hiddenNodes1.toDouble().pow(-0.5), hiddenNodes2, hiddenNodes1)
        wHiddenOutput = mk.rand(from = -hiddenNodes2.toDouble().pow(-0.5),
            until = hiddenNodes2.toDouble().pow(-0.5), outputNodes, hiddenNodes2)

    }

    fun query(inputsList : MutableList<Double>) : List<Double> {
        val inputs = mk.ndarray(inputsList).reshape(1, inputsList.size).transpose()

        val hidden1Inputs = mk.linalg.dot(wInputHidden, inputs)
        val hidden1Outputs = activationFun(hidden1Inputs)

        val hidden2Inputs = mk.linalg.dot(wHiddenHidden, hidden1Outputs)
        val hidden2Outputs = activationFun(hidden2Inputs)

        val finalInputs = mk.linalg.dot(wHiddenOutput, hidden2Outputs)
        val finalOutputs = activationFun(finalInputs)

        return finalOutputs.toList()
    }

    fun train(inputsList: MutableList<Double>, targetList: MutableList<Double>) {
        val inputs = mk.ndarray(inputsList).reshape(1,inputsList.size).transpose()
        val targets = mk.ndarray(targetList).reshape(1,targetList.size).transpose()

        val hidden1Inputs = mk.linalg.dot(wInputHidden, inputs)
        val hidden1Outputs = activationFun(hidden1Inputs)

        val hidden2Inputs = mk.linalg.dot(wHiddenHidden, hidden1Outputs)
        val hidden2Outputs = activationFun(hidden2Inputs)

        val finalInputs = mk.linalg.dot(wHiddenOutput, hidden2Outputs)
        val finalOutputs = activationFun(finalInputs)

        val outputErrors = targets - finalOutputs
        val hidden2Errors = mk.linalg.dot(wHiddenOutput.transpose(), outputErrors)
        val hidden1Errors = mk.linalg.dot(wHiddenHidden.transpose(), hidden2Errors)

        wHiddenOutput = correctedWeights(wHiddenOutput, finalOutputs, hidden2Outputs, outputErrors, learningRate)
        wHiddenHidden = correctedWeights(wHiddenHidden, hidden2Outputs, hidden1Outputs, hidden2Errors, learningRate)
        wInputHidden = correctedWeights(wInputHidden,hidden1Outputs, inputs, hidden1Errors, learningRate)
    }

    fun getWeights() : List<List<DoubleArray>> {
        val firstLayerWeights = mutableListOf<DoubleArray>()
        val secondLayerWeights = mutableListOf<DoubleArray>()
        val thirdLayerWeights = mutableListOf<DoubleArray>()
        for(i in 0 until wInputHidden.toListD2().size) {
            firstLayerWeights += wInputHidden.toListD2()[i].toDoubleArray()
        }
        for(i in 0 until wHiddenHidden.toListD2().size) {
            secondLayerWeights += wHiddenHidden.toListD2()[i].toDoubleArray()
        }
        for(i in 0 until wHiddenOutput.toListD2().size) {
            thirdLayerWeights += wHiddenOutput.toListD2()[i].toDoubleArray()
        }
        return listOf(firstLayerWeights, secondLayerWeights, thirdLayerWeights)
    }

    fun setWeights(weights: List<List<DoubleArray>>) {
        for (i in 0 until weights[0].size) {
            wInputHidden[i] = mk.ndarray(weights[0][i])
        }
        for (i in 0 until weights[1].size) {
            wHiddenHidden[i] = mk.ndarray(weights[1][i])
        }
        for (i in 0 until weights[2].size) {
            wHiddenOutput[i] = mk.ndarray(weights[2][i])
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

private fun correctedWeights(oldWeights: NDArray<Double, D2>,
                             outputsOfLayer: NDArray<Double, D2>,
                             outputsOfPreviousLayer: NDArray<Double, D2>,
                             layerErrors: NDArray<Double, D2>,
                             learningRate: Double) : NDArray<Double, D2> {
    return oldWeights + (learningRate * (mk.linalg.dot((layerErrors * outputsOfLayer *
            (1.0 - outputsOfLayer)), outputsOfPreviousLayer.transpose())))
}