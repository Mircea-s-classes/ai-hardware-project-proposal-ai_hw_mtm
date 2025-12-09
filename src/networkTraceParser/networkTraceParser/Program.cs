// See https://aka.ms/new-console-template for more information

using System.Globalization;
using CsvHelper;
using networkTraceParser.Models;

Console.WriteLine("Hello, World!");
using var reader = new StreamReader("../../../digital.csv");
using var csv = new CsvReader(reader,  CultureInfo.InvariantCulture);
csv.Context.RegisterClassMap<DataInstanceMap>();
var records = csv.GetRecords<DataInstance>().ToList();
Console.WriteLine(records.Count);
var bits = records
    .Zip(records.Skip(1), (prev, curr) => new { prev, curr }) 
    .Where(pair => pair.prev.Channel3 == 0 && pair.curr.Channel3 == 1)
    .Select(pair => new Bit(pair.prev, pair.curr))
    .ToList();

var packets = new List<Packet>();
Packet? currentPacket = null;
var consecutiveLowCycles = 0;

for (var i = 0; i < records.Count; i++)
{
    var current = records[i];
    
    if (current.Channel7 == 0)
    {
        consecutiveLowCycles++;

        // STOP CONDITION: Clock is 0 for at least 5 cycles
        // If we have an active packet and hit 5 lows, close it.
        if (currentPacket != null && consecutiveLowCycles >= 5)
        {
            currentPacket.EndInstance = current;
            packets.Add(currentPacket);
            currentPacket = null; // Stop collecting
        }
    }
    else // Channel 3 is 1 (High)
    {
        // If we were low previously, this is a Rising Edge
        if (consecutiveLowCycles > 0)
        {
            var prior = records[i - 1]; // Safe because lowCycles > 0 means i > 0

            // START CONDITION: Clock was 0 for EXACTLY 3 cycles
            if (currentPacket == null && consecutiveLowCycles == 3)
            {
                // Start a new packet
                currentPacket = new Packet(current, current); 
                // Note: We intentionally pass 'current' as end temporarily to avoid nulls
            }
            
            // BIT COLLECTION: If we are inside a valid packet, record the bit
            // This captures the bit on the rising edge of the oscillation
            if (currentPacket != null)
            {
                currentPacket.Bits.Add(new Bit(prior, current));
            }
        }

        // Reset the low counter because the clock is now High
        consecutiveLowCycles = 0;
    }
}

Console.WriteLine(bits.Count);

Console.WriteLine($"Found {packets.Count} packets.");

// Example: Print bits of the first packet to verify
if (packets.Any())
{
    foreach (var firstPacketValues in packets.Select(packet => packet.Bits.Select(b => b.Value ? "1" : "0")))
    {
        Console.WriteLine("Packet 1 Data: " + string.Join("", firstPacketValues));
    }
}