namespace networkTraceParser.Models;

public class Packet (DataInstance StartInstance,  DataInstance EndInstance)
{
    public List<Bit> Bits { get; set; } = [];
    public DataInstance StartInstance { get; set; } =  StartInstance;
    public DataInstance EndInstance { get; set; } =  EndInstance;
}