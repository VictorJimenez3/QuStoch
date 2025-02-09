// app/page.tsx
export default function Home() {
  const stats = ['Price', 'Volume', 'Market Cap']

  return (
    <main className="min-h-screen p-8">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* Search Section */}
        <div className="space-y-4">
          <h2 className="text-2xl text-white font-semibold">
            Search Stock Analytics
          </h2>
          <div className="relative">
            <input
              type="text"
              placeholder="Search a stock..."
              className="w-full p-4 rounded-lg bg-capital-one-light-gray/10 
                       text-white placeholder-capital-one-light-gray
                       focus:outline-none focus:ring-2 focus:ring-capital-one-red
                       transition-all duration-200"
            />
          </div>
        </div>

        {/* Graph Placeholder */}
        <div className="bg-capital-one-dark-blue/50 rounded-xl p-6 
                      border border-capital-one-light-gray/20
                      backdrop-blur-sm">
          <div className="h-96 bg-capital-one-dark-blue/30 rounded-lg 
                        flex items-center justify-center text-capital-one-light-gray
                        animate-pulse">
            Graph Visualization (To Be Implemented)
          </div>
        </div>

        {/* Stats Grid Placeholder */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {stats.map((stat) => (
            <div 
              key={stat}
              className="bg-capital-one-dark-blue/50 p-6 rounded-xl 
                       border border-capital-one-light-gray/20
                       hover:bg-capital-one-dark-blue/70 transition-colors
                       backdrop-blur-sm"
            >
              <h3 className="text-capital-one-light-gray text-lg mb-2">{stat}</h3>
              <div className="text-2xl text-white font-bold">--</div>
            </div>
          ))}
        </div>
      </div>
    </main>
  )
}