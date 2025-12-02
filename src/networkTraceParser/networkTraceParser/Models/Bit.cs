namespace networkTraceParser.Models;

public class Bit(DataInstance priorInstance, DataInstance nextInstance)
{
    public DataInstance PriorInstance { get; set; } = priorInstance;
    public DataInstance NextInstance { get; set; } = nextInstance;
    public bool Value { get; set; } = priorInstance.Channel2 == 0;
}