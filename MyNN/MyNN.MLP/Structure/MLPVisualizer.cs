using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using MyNN.MLP.Structure.Layer;
using MyNN.MLP.Structure.Neuron;

namespace MyNN.MLP.Structure
{
    public class MLPVisualizer
    {
        private readonly IMLP _mlp;

        public MLPVisualizer(
            IMLP mlp)
        {
            if (mlp == null)
            {
                throw new ArgumentNullException("mlp");
            }

            _mlp = mlp;
        }

        public Bitmap GetVisualScheme()
        {
            const int imageWidth = 2000;
            const int imageHeight = 2000;
            const int neuronDiameter = 50;
            const int neuronRadius = 25;

            var result = new Bitmap(imageWidth, imageHeight);
            using (var g = Graphics.FromImage(result))
            {
                g.CompositingQuality = CompositingQuality.HighQuality;
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;

                g.Clear(Color.LightGreen);

                if (_mlp.Layers.Length <= 5 && _mlp.Layers.All(j => j.TotalNeuronCount < 50))
                {
                    // ------------------------- DRAW NEURONS -------------------------
                    using (var neuronFont = new Font("Times New Roman", 12, FontStyle.Regular))
                    {
                        for (var layerIndex = 0; layerIndex < _mlp.Layers.Length; layerIndex++)
                        {
                            var l = _mlp.Layers[layerIndex];

                            for (int neuronIndex = 0, neuronCount = l.TotalNeuronCount; neuronIndex < neuronCount; neuronIndex++)
                            {
                                var n = l.Neurons[neuronIndex];

                                using (var neuronColor = GetNeuronColor(layerIndex, neuronCount, neuronIndex, n))
                                {
                                    int x;
                                    int y;
                                    GetNeuronCenterPosition(
                                        imageWidth,
                                        imageHeight,
                                        _mlp.Layers.Length,
                                        layerIndex,
                                        neuronCount,
                                        neuronIndex,
                                        neuronRadius,
                                        out x,
                                        out y
                                        );

                                    g.DrawEllipse(
                                        neuronColor,
                                        x - neuronRadius,
                                        y - neuronRadius,
                                        neuronDiameter,
                                        neuronDiameter);

                                    g.DrawString(
                                        l.LayerActivationFunction.ShortName,
                                        neuronFont,
                                        neuronColor.Brush,
                                        x - (neuronDiameter * 0.4f),
                                        y - (neuronDiameter * 0.2f));
                                }
                            }
                        }
                    }

                    // ------------------------- DRAW WEIGHTS -------------------------

                    using (var weightFont = new Font("Times New Roman", 12, FontStyle.Regular))
                    {
                        for (var layerIndex = 0; layerIndex < _mlp.Layers.Length; layerIndex++)
                        {
                            var l = _mlp.Layers[layerIndex];

                            var allowedLayerTypes = new List<LayerTypeEnum>
                            {
                                LayerTypeEnum.Input,
                                LayerTypeEnum.FullConnected
                            };

                            if (!allowedLayerTypes.Contains(l.Type))
                            {
                                throw new NotSupportedException(
                                    string.Format(
                                        "Данный визуалайзер поддержиает только слои типа: {0}",
                                        string.Join(",", allowedLayerTypes)
                                    ));
                            }

                            for (int neuronIndex = 0, neuronCount = l.TotalNeuronCount; neuronIndex < neuronCount; neuronIndex++)
                            {
                                var n = l.Neurons[neuronIndex];

                                for (int weightIndex = 0, weightCount = n.Weights.Length; weightIndex < weightCount; weightIndex++)
                                {
                                    using (var weightColor = GetWeightColor())
                                    {

                                        int xleft;
                                        int yleft;
                                        GetNeuronCenterPosition(
                                            imageWidth,
                                            imageHeight,
                                            _mlp.Layers.Length,
                                            layerIndex - 1,
                                            _mlp.Layers[layerIndex - 1].TotalNeuronCount,
                                            weightIndex,
                                            neuronRadius,
                                            out xleft,
                                            out yleft
                                            );

                                        int xright;
                                        int yright;
                                        GetNeuronCenterPosition(
                                            imageWidth,
                                            imageHeight,
                                            _mlp.Layers.Length,
                                            layerIndex,
                                            neuronCount,
                                            neuronIndex,
                                            neuronRadius,
                                            out xright,
                                            out yright
                                            );

                                        var angle = Math.Atan2(yright - yleft, xright - xleft);

                                        var xleft2 = xleft + (int)(neuronRadius * Math.Cos(angle));
                                        var yleft2 = yleft + (int)(neuronRadius * Math.Sin(angle));

                                        var xright2 = xright - (int)(neuronRadius * Math.Cos(angle));
                                        var yright2 = yright - (int)(neuronRadius * Math.Sin(angle));

                                        g.DrawLine(
                                            weightColor,
                                            xleft2,
                                            yleft2,
                                            xright2,
                                            yright2);

                                        var xtext = xright - (int)((4 * neuronRadius) * Math.Cos(angle));
                                        var ytext = yright - (int)((4 * neuronRadius) * Math.Sin(angle));

                                        g.DrawString(
                                            n.Weights[weightIndex].ToString(),
                                            weightFont,
                                            weightColor.Brush,
                                            xtext - 40,
                                            ytext - 5);

                                    }
                                }
                            }
                        }
                    }
                }

            }

            return result;
        }

        private Pen GetWeightColor(
            )
        {
            Pen result = null;

            result = new Pen(Brushes.Black);

            return result;
        }

        private Pen GetNeuronColor(
            int layerIndex,
            int neuronCount,
            int neuronIndex,
            INeuron neuron
            )
        {
            var result = layerIndex == 0 ? new Pen(Brushes.Red) : new Pen(Brushes.Green);

            return result;
        }

        private void GetNeuronCenterPosition(
            int imageWidth,
            int imageHeight,
            int layerCount,
            int layerIndex,
            int neuronCount,
            int neuronIndex,
            int neuronRadius,
            out int x,
            out int y)
        {
            var wStep = imageWidth / layerCount;
            var leftShift = wStep / 2;

            var hStep = imageHeight / neuronCount;
            var topShift = hStep / 2;

            x = leftShift + wStep * layerIndex + neuronRadius;
            y = topShift + hStep * neuronIndex + neuronRadius;
        }

    }

}
