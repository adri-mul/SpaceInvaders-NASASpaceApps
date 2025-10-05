import { useState } from 'react';
import { Rocket, Globe2, Activity } from 'lucide-react';
import './home.css';
import MapPage from './SpaceMap3D'; // your 3D space map component

// Example dataset: 30 planets
const planets = [
  { name: 'Kepler-22b', radius: '2.4 R⊕', period: '289.9 days', distance: '600 ly' },
  { name: 'Kepler-452b', radius: '1.63 R⊕', period: '384.8 days', distance: '1400 ly' },
  { name: 'Kepler-186f', radius: '1.1 R⊕', period: '129.9 days', distance: '490 ly' },
  { name: 'TOI-700d', radius: '1.19 R⊕', period: '37.4 days', distance: '101.5 ly' },
  { name: 'K2-18b', radius: '2.6 R⊕', period: '32.9 days', distance: '124 ly' },
  { name: 'Kepler-62f', radius: '1.41 R⊕', period: '267.3 days', distance: '1200 ly' },
  { name: 'Kepler-69c', radius: '1.7 R⊕', period: '242.5 days', distance: '2700 ly' },
  { name: 'Kepler-442b', radius: '1.34 R⊕', period: '112.3 days', distance: '1200 ly' },
  { name: 'Kepler-22c', radius: '1.8 R⊕', period: '350.0 days', distance: '600 ly' },
  { name: 'Kepler-62e', radius: '1.61 R⊕', period: '122.4 days', distance: '1200 ly' },
  { name: 'Kepler-186e', radius: '1.29 R⊕', period: '129.9 days', distance: '490 ly' },
  { name: 'TOI-1231b', radius: '3.7 R⊕', period: '24.25 days', distance: '90 ly' },
  { name: 'K2-3d', radius: '1.5 R⊕', period: '44.6 days', distance: '140 ly' },
  { name: 'Kepler-138b', radius: '0.57 R⊕', period: '10.3 days', distance: '200 ly' },
  { name: 'Kepler-138c', radius: '0.95 R⊕', period: '13.8 days', distance: '200 ly' },
  { name: 'Kepler-138d', radius: '1.2 R⊕', period: '23.1 days', distance: '200 ly' },
  { name: 'Kepler-440b', radius: '1.86 R⊕', period: '101.1 days', distance: '320 ly' },
  { name: 'Kepler-441b', radius: '1.87 R⊕', period: '203.0 days', distance: '420 ly' },
  { name: 'Kepler-443b', radius: '2.33 R⊕', period: '177.5 days', distance: '2500 ly' },
  { name: 'Kepler-444f', radius: '0.54 R⊕', period: '9.7 days', distance: '116 ly' },
  { name: 'Kepler-62c', radius: '0.54 R⊕', period: '12.4 days', distance: '1200 ly' },
  { name: 'Kepler-22d', radius: '2.0 R⊕', period: '400.0 days', distance: '600 ly' },
  { name: 'Kepler-452c', radius: '1.9 R⊕', period: '390.0 days', distance: '1400 ly' },
  { name: 'Kepler-186g', radius: '1.2 R⊕', period: '150.0 days', distance: '490 ly' },
  { name: 'TOI-700b', radius: '1.04 R⊕', period: '9.9 days', distance: '101.5 ly' },
  { name: 'TOI-700c', radius: '1.12 R⊕', period: '16.0 days', distance: '101.5 ly' },
  { name: 'K2-18c', radius: '2.3 R⊕', period: '8.96 days', distance: '124 ly' },
  { name: 'K2-72e', radius: '1.29 R⊕', period: '24.2 days', distance: '160 ly' },
  { name: 'K2-72d', radius: '1.03 R⊕', period: '15.7 days', distance: '160 ly' },
  { name: 'K2-72c', radius: '1.10 R⊕', period: '7.8 days', distance: '160 ly' },
];

// Planet cards component
function PlanetCardList() {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 mb-10">
      {planets.map((planet, idx) => (
        <div
          key={idx}
          className="bg-zinc-800/70 backdrop-blur-md rounded-2xl border border-zinc-700/50 shadow-lg hover:scale-105 transform transition-all duration-300 flex flex-col justify-center items-center p-5 aspect-square"
        >
          <h3 className="text-xl font-bold text-white mb-2 text-center">{planet.name}</h3>
          <p className="text-zinc-400 text-sm text-center"><span className="font-semibold">Radius:</span> {planet.radius}</p>
          <p className="text-zinc-400 text-sm text-center"><span className="font-semibold">Orbital:</span> {planet.period}</p>
          <p className="text-zinc-400 text-sm text-center"><span className="font-semibold">Distance:</span> {planet.distance}</p>
        </div>
      ))}
    </div>
  );
}

function Home() {
  const [activeTab, setActiveTab] = useState<'home' | 'map'>('home');

  return (
    <div className="min-h-screen bg-zinc-950 relative overflow-hidden">
      {/* Background animation */}
      <div className="absolute inset-0 bg-gradient-to-br from-zinc-950 via-zinc-900 to-zinc-950">
        <div className="absolute inset-0 opacity-20" style={{
          backgroundImage: 'linear-gradient(rgba(127,127,127,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(127,127,127,0.1) 1px, transparent 1px)',
          backgroundSize: '50px 50px'
        }} />
        {[...Array(40)].map((_, i) => (
          <div key={i} className="absolute bg-zinc-400 rounded-full" style={{
            width: Math.random() * 2 + 0.5 + 'px',
            height: Math.random() * 2 + 0.5 + 'px',
            top: Math.random() * 100 + '%',
            left: Math.random() * 100 + '%',
            opacity: Math.random() * 0.5 + 0.2,
          }} />
        ))}
        <div className="absolute top-0 right-1/4 w-96 h-96 bg-red-600/10 rounded-full blur-3xl" />
        <div className="absolute bottom-0 left-1/4 w-96 h-96 bg-red-500/5 rounded-full blur-3xl" />
      </div>

      {/* Navbar */}
      <nav className="bg-zinc-900/60 backdrop-blur-xl border-b border-zinc-800/50 relative z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex items-center justify-between h-20">
          <div className="flex items-center space-x-3">
            <div className="relative">
              <div className="w-10 h-10 bg-gradient-to-br from-red-600 to-red-500 rounded-lg flex items-center justify-center shadow-lg shadow-red-500/20">
                <Rocket className="w-6 h-6 text-white" />
              </div>
              <div className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full animate-pulse" />
            </div>
            <div>
              <div className="text-xl font-bold text-white tracking-tight">NEXUS</div>
              <div className="text-xs text-zinc-500 tracking-wider">SPACE SYSTEMS</div>
            </div>
          </div>

          {/* Tabs */}
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setActiveTab('home')}
              className={`group relative px-6 py-2.5 rounded-lg font-medium text-sm transition-all duration-300 ${
                activeTab === 'home'
                  ? 'bg-gradient-to-r from-red-600 to-red-500 text-white shadow-lg shadow-red-500/30'
                  : 'bg-zinc-800/40 text-zinc-400 hover:text-white hover:bg-zinc-800 border border-zinc-700/50 hover:border-zinc-600'
              }`}
            >
              <Activity className="w-4 h-4 mr-2 inline" /> Home
            </button>
            <button
              onClick={() => setActiveTab('map')}
              className={`group relative px-6 py-2.5 rounded-lg font-medium text-sm transition-all duration-300 ${
                activeTab === 'map'
                  ? 'bg-gradient-to-r from-red-600 to-red-500 text-white shadow-lg shadow-red-500/30'
                  : 'bg-zinc-800/40 text-zinc-400 hover:text-white hover:bg-zinc-800 border border-zinc-700/50 hover:border-zinc-600'
              }`}
            >
              <Globe2 className="w-4 h-4 mr-2 inline" /> Map
            </button>
          </div>
        </div>
      </nav>

      {/* Content */}
      <main className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {activeTab === 'home' && (
          <>
            <PlanetCardList />
            <div className="relative bg-gradient-to-br from-zinc-900/80 to-zinc-900/60 backdrop-blur-xl rounded-2xl p-10 border border-zinc-800/50 shadow-2xl">
              <h1 className="text-5xl font-bold text-white mb-3 tracking-tight">Command Center</h1>
              <p className="text-zinc-400 text-lg leading-relaxed max-w-2xl">MINIMAL AI WAS USED FOR THIS PROJECT, JUST TO OPTIMIZE CODE AT THE END, AND RESEARCH!!<br /><br />
              About Us:<br />
              ArjunS<br />
              AvinashK<br />
              AdyaS<br />
              KrishD<br />
              Team Captian: AdrianM<br /><br />
              Custom Neural Networks Created: 5<br /><br />
              Project Space Invaders uses an AI model to perform a variety of functions relating to data analytics and prediction. Trained on NASA data sets, our python based AI uses raw NASA light curve data to extract information, which it then uses to make predictions regarding possibilities of an exoplanet. Additionally, our project models over 12000 exoplanet objects in 3d space, and includes information sourced using SerpAPI. We can save time and money for NASA scientists by automating the data sorting and prediction. By integrating data analytics and prediction, project Space Invaders becomes a unique, versatile, and efficient tool for detecting and modeling exoplanets.<br /><br />
              NASA Data<br />
              Kepler Objects of Interest (KOI)<br />
              TESS Objects of Interest (TOI)<br />
              K2 Planets and Candidates<br />
              NASA Light Curve Data<br />
              BULK NASA Light Curve Data<br />
              Light Curve DATA Space Telescope Science Institute (STScI) on behalf of NASA<br />
              </p>
            </div>
          </>
        )}

        {activeTab === 'map' && (
          <div className="w-full h-[80vh] rounded-2xl overflow-hidden border border-zinc-800/50 shadow-2xl">
            <MapPage />
          </div>
        )}
      </main>
    </div>
  );
}

export default Home;
