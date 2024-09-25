"use client";

import { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Search, X, TrendingUp, RefreshCcw } from "lucide-react";
import { PolarAngleAxis, PolarGrid, Radar, RadarChart, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, LabelList } from "recharts";
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import '../scrollbar.css';
import '../buttons.css';

const apiUrl = import.meta.env.VITE_API_BASE_URL;

interface ScoringCategory {
    name: string;
    label: string;
    defaultValue: number;
}

interface Player {
    rank?: number;
    player_id: number;
    player_name: string;
    fantasy_score?: number;
    position: string;
    MIN: number;
    PTS: number;
    AST: number;
    REB: number;
    STL: number;
    BLK: number;
    TO: number;
    FGA: number;
    FGM: number;
    FTA: number;
    FTM: number;
    FG3M: number;
    [key: string]: any;
}

interface ScoringSystem {
    [key: string]: number;
}

const espnPreset: ScoringSystem = {
    PTS: 1,
    AST: 1,
    REB: 1,
    STL: 2,
    BLK: 2,
    TOV: -1,
    FGM: 1,
    FGA: -1,
    FTM: 1,
    FTA: -1,
    FG3M: 1,
};

const sleeperPreset: ScoringSystem = {
    PTS: 1,
    AST: 1.5,
    REB: 1.2,
    STL: 3,
    BLK: 3,
    TOV: -1,
    FGM: 1,
    FGA: -1,
    FTM: 1,
    FTA: -1,
    FG3M: 1,
};

const yahooPreset: ScoringSystem = {
    PTS: 1,
    AST: 1.5,
    REB: 1.2,
    STL: 3,
    BLK: 3,
    TOV: -1,
    FGM: 1,
    FGA: -1,
    FTM: 1,
    FTA: -1,
    FG3M: 0.5,
};

const fanduelPreset: ScoringSystem = {
    PTS: 1,
    AST: 1.5,
    REB: 1.5,
    STL: 3,
    BLK: 3,
    TOV: -2,
    FGM: 1.5,
    FGA: -1,
    FTM: 1,
    FTA: -1,
    FG3M: 1,
};

const nbaOfficialPreset: ScoringSystem = {
    PTS: 1,
    AST: 1.5,
    REB: 1.2,
    STL: 3,
    BLK: 3,
    TOV: -1,
    FGM: 0,
    FGA: 0,
    FTM: 0,
    FTA: 0,
    FG3M: 0,
};

const scoringCategories: ScoringCategory[] = [
    { name: 'PTS', label: 'Points (PTS)', defaultValue: 1 },
    { name: 'AST', label: 'Assists (AST)', defaultValue: 1 },
    { name: 'REB', label: 'Rebounds (REB)', defaultValue: 1 },
    { name: 'STL', label: 'Steals (STL)', defaultValue: 1 },
    { name: 'BLK', label: 'Blocks (BLK)', defaultValue: 1 },
    { name: 'TOV', label: 'Turnovers (TOV)', defaultValue: -1 },
    { name: 'FGM', label: 'Field Goals Made (FGM)', defaultValue: 1 },
    { name: 'FGA', label: 'Field Goals Attempted (FGA)', defaultValue: -1 },
    { name: 'FTM', label: 'Free Throws Made (FTM)', defaultValue: 1 },
    { name: 'FTA', label: 'Free Throws Attempted (FTA)', defaultValue: -1 },
    { name: 'FG3M', label: 'Three Pointers Made (3PM)', defaultValue: 1 },
]

const positions: string[] = ['All', 'PG', 'SG', 'SF', 'PF', 'C']

type Position = 'PG' | 'SG' | 'SF' | 'PF' | 'C';
const positionColors: Record<Position, string> = {
    'PG': 'hsl(210, 100%, 80%)',
    'SG': 'hsl(210, 100%, 65%)',
    'SF': 'hsl(210, 100%, 50%)',
    'PF': 'hsl(210, 100%, 35%)',
    'C': 'hsl(210, 100%, 15%)'
};

const getColorForPosition = (position: string): string => {
    const positions = position.split('-') as Position[];
    const primaryPosition = positions[0];
    return primaryPosition in positionColors ? positionColors[primaryPosition] : 'hsl(210, 100%, 50%)';
};

const barChartConfig = {
    fantasy_score: {
        label: "Fantasy Score",
        color: "hsl(var(--chart-1))",
    },
    label: {
        color: "#D1D5DB",
    },
};

export default function Dashboard() {
    const [scoringSystem, setScoringSystem] = useState<ScoringSystem>(() => {
        const savedSystem = localStorage.getItem('scoringSystem');
        return savedSystem ? JSON.parse(savedSystem) : {};
    });
    const [players, setPlayers] = useState<Player[]>(() => {
        const savedPlayers = localStorage.getItem('players');
        return savedPlayers ? JSON.parse(savedPlayers) : [];
    });
    const [removedPlayerIds, setRemovedPlayerIds] = useState<number[]>(() => {
        const savedRemovedIds = localStorage.getItem('removedPlayerIds');
        return savedRemovedIds ? JSON.parse(savedRemovedIds) : [];
    });
    const [filteredPlayers, setFilteredPlayers] = useState<Player[]>([]);
    const [selectedPosition, setSelectedPosition] = useState<string>('All');
    const [selectedPlayer, setSelectedPlayer] = useState<Player | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [searchTerm, setSearchTerm] = useState<string>('');
    const [isUpdating, setIsUpdating] = useState<boolean>(false);
    const [isDialogOpen, setIsDialogOpen] = useState(false);
    const [topPlayers, setTopPlayers] = useState<Player[]>([]);
    const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);

    const fetchPlayers = useCallback(async (): Promise<Player[]> => {
        try {
            const response = await fetch(`${apiUrl}/api/players`);
    
            if (!response.ok) {
                throw new Error(`Failed to fetch players. Status: ${response.status}`);
            }
    
            const contentType = response.headers.get('content-type');
            if (!contentType || !contentType.includes('application/json')) {
                throw new Error('Invalid response: Expected JSON');
            }
    
            const data: Player[] = await response.json();
            const filteredData = data.filter(player => !removedPlayerIds.includes(player.player_id));
            setPlayers(filteredData);
            localStorage.setItem('players', JSON.stringify(filteredData));
            setError(null);
            return filteredData;
        } catch (error) {
            console.error('Error fetching players:', error);
            setError('Failed to fetch players. Please try again later.');
            return [];
        }
    }, [removedPlayerIds]);
    
    useEffect(() => {
        const initialSystem: ScoringSystem = {};
        scoringCategories.forEach(category => {
            initialSystem[category.name] = category.defaultValue;
        });
        if (Object.keys(scoringSystem).length === 0) {
            setScoringSystem(initialSystem);
            localStorage.setItem('scoringSystem', JSON.stringify(initialSystem));
        }
    }, [scoringSystem]);

    useEffect(() => {
        const initialiseData = async () => {
            if (players.length === 0) {
                await fetchPlayers();
            }
        };
        initialiseData();
    }, [fetchPlayers, players.length]);

    const prepareTopPlayersData = (players: Player[]): Player[] => {
        return players
            .filter(player => player.fantasy_score !== undefined)
            .sort((a, b) => (b.fantasy_score || 0) - (a.fantasy_score || 0))
            .slice(0, 20);
    };

    const filterPlayers = useCallback((): void => {
        let filtered = [...players];
        if (selectedPosition !== 'All') {
            filtered = filtered.filter(player => {
                const playerPositions = player.position.split('-');
                return playerPositions.includes(selectedPosition);
            });
        }
        if (searchTerm) {
            filtered = filtered.filter(player =>
                player.player_name.toLowerCase().includes(searchTerm.toLowerCase())
            );
        }
        setFilteredPlayers(filtered);
        setTopPlayers(prepareTopPlayersData(filtered));
    }, [players, selectedPosition, searchTerm]);

    useEffect(() => {
        filterPlayers();
    }, [filterPlayers]);

    const applyPreset = async (preset: ScoringSystem) => {
        setScoringSystem(preset);
        setHasUnsavedChanges(true);
    };

    const getDisplayValue = (categoryName: string, value: number) => {
        if (value === 0) return "0.0";
        const category = scoringCategories.find(cat => cat.name === categoryName);
        return (value || category?.defaultValue || 0).toFixed(1);
    };

    const PositionLegend = () => (
        <div className="flex justify-center gap-4 mt-4 text-gray-400">
            {Object.entries(positionColors).map(([position, color]) => (
                <div key={position} className="flex items-center">
                    <div className="w-4 h-4 mr-2 rounded" style={{ backgroundColor: color }}></div>
                    <span>{position}</span>
                </div>
            ))}
        </div>
    );

    const HorizontalBarChart: React.FC<{ data: Player[] }> = ({ data }) => {
        const [chartHeight, setChartHeight] = useState<number>(500);
        const [playersToShow, setPlayersToShow] = useState<number>(20);
        const [playerNameFontSize, setPlayerNameFontSize] = useState<number>(12);
        const [yAxisWidth, setYAxisWidth] = useState<number>(100);

        useEffect(() => {
            const handleResize = () => {
                const width = window.innerWidth;
                if (width < 640) { // Mobile
                    setChartHeight(175);
                    setPlayersToShow(8);
                    setPlayerNameFontSize(10);
                    setYAxisWidth(120);
                } else if (width < 1500) { // Laptop
                    setChartHeight(650);
                    setPlayersToShow(20);
                    setPlayerNameFontSize(12);
                    setYAxisWidth(185);
                } else { // Desktop
                    setChartHeight(650);
                    setPlayersToShow(20);
                    setPlayerNameFontSize(12);
                    setYAxisWidth(185);
                }
            };
            handleResize(); // Call once to set initial size
            window.addEventListener('resize', handleResize);
            return () => window.removeEventListener('resize', handleResize);
        }, []);

        const chartData = data.slice(0, playersToShow);

        const CustomBar = (props: any) => {
            const { x, y, width, height, payload } = props;
            const fill = getColorForPosition(payload.position);

            return (
                <rect
                    x={x}
                    y={y}
                    width={width}
                    height={height}
                    fill={fill}
                    rx={4}
                    ry={4}
                    onClick={() => {
                        setSelectedPlayer(payload);
                        setIsDialogOpen(true);
                    }}
                    style={{ cursor: 'pointer' }}
                />
            );
        };

        return (
            <div style={{ width: '100%', height: chartHeight }}>
                {/* ChartContainer has left to right animation but no dynamic height resizing */}
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                        data={chartData}
                        layout="vertical"
                        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                        <CartesianGrid horizontal={false} />
                        <YAxis
                            dataKey="player_name"
                            type="category"
                            width={yAxisWidth}
                            tickLine={false}
                            axisLine={false}
                            tick={{ 
                                fill: 'var(--color-label)', 
                                fontSize: playerNameFontSize,
                                width: yAxisWidth - 10, // Adjust this value as needed
                                overflow: 'hidden',
                            }}
                        />
                        <XAxis type="number" hide />
                        <Bar
                            dataKey="fantasy_score"
                            shape={<CustomBar />}
                            radius={[4, 4, 0, 0]}
                            
                        >
                            <LabelList
                                dataKey="fantasy_score"
                                position="right"
                                offset={8}
                                className="fill-foreground"
                                fontSize={playerNameFontSize}
                                formatter={(value: number) => value.toFixed(2)}
                            />
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
            </div>
        );
    };

    const updatePlayerScores = async (setLoadingState: boolean = true): Promise<void> => {
        if (setLoadingState) setIsUpdating(true);
        try {
            const response = await fetch(`${apiUrl}/api/score`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ scoring_system: scoringSystem }),
            });
            if (!response.ok) {
                throw new Error('Failed to update player scores');
            }
            const newScores: Player[] = await response.json();
            // console.log(newScores)

            const playersToSort = newScores.length > 0 ? newScores : players;
            const sortedPlayers = playersToSort.sort((a, b) => {
                if (a.fantasy_score === undefined) return 1;
                if (b.fantasy_score === undefined) return -1;
                return b.fantasy_score - a.fantasy_score;
            }).map((player, index) => ({
                ...player,
                rank: index + 1
            }));

            setPlayers(sortedPlayers);
            setError(null);
        } catch (error) {
            console.error('Error updating player scores:', error);
            setError('Failed to update player scores. Please try again later.');
        } finally {
            if (setLoadingState) setIsUpdating(false);
        }
    };

    const handleScoringChange = (category: string, value: number): void => {
        const newScoringSystem = { ...scoringSystem, [category]: value };
        setScoringSystem(newScoringSystem);
        setHasUnsavedChanges(true);
    };

    const handleUpdateScores = () => {
        updatePlayerScores(true);
        setHasUnsavedChanges(false);
        localStorage.setItem('scoringSystem', JSON.stringify(scoringSystem));
        localStorage.removeItem('removedPlayerIds');
        setRemovedPlayerIds([]);
    };

    const handlePlayerClick = (player: Player): void => {
        setSelectedPlayer(player);
        setIsDialogOpen(true);
    };

    const handleRemovePlayer = (playerId: number) => {
        const newPlayers = players.filter(p => p.player_id !== playerId);
        setPlayers(newPlayers);
        localStorage.setItem('players', JSON.stringify(newPlayers));
        
        const newRemovedIds = [...removedPlayerIds, playerId];
        setRemovedPlayerIds(newRemovedIds);
        localStorage.setItem('removedPlayerIds', JSON.stringify(newRemovedIds));
        
        setFilteredPlayers(filteredPlayers.filter(p => p.player_id !== playerId));
    };

    const handleResetAllData = () => {
        localStorage.removeItem('scoringSystem');
        localStorage.removeItem('players');
        localStorage.removeItem('removedPlayerIds');
        setScoringSystem({});
        setPlayers([]);
        setRemovedPlayerIds([]);
        setFilteredPlayers([]);
        setTopPlayers([]);
        setHasUnsavedChanges(false);
        fetchPlayers();
    };

    const renderPlayerStats = (inDialog: boolean = false) => {
        if (!selectedPlayer) return <p className="text-gray-300 text-center text-muted-foreground">Select a player to view their stats</p>

        const chartData = scoringCategories.map(category => {
            let value;
            if (category.name === 'REB') {
                value = ((selectedPlayer['predicted_OREB'] || 0) + (selectedPlayer['predicted_DREB'] || 0)) * (scoringSystem['REB'] || 1);
            } else {
                value = (selectedPlayer[`predicted_${category.name}`] || 0) * (scoringSystem[category.name] || 1);
            }
            return {
                category: category.name,
                value: value,
            };
        });

        const chartConfig = {
            x: {
                label: 'Category',
                theme: { light: 'hsl(var(--primary))', dark: 'hsl(var(--primary))' },
            },
            y: {
                label: 'Value',
                theme: { light: 'hsl(var(--primary))', dark: 'hsl(var(--primary))' },
            },
            tooltip: {
                label: 'Value',
                icon: () => null,
                color: 'hsl(var(--primary))',
            },
        }

        const content = (
            <>
                <CardHeader className="items-center pb-2">
                    <CardTitle className="text-gray-300">
                        {selectedPlayer.player_name}
                    </CardTitle>
                    <CardDescription className="text-gray-300">
                        <div>
                            <span>Fantasy Points: {selectedPlayer.fantasy_score?.toFixed(2) || 'N/A'}</span>
                        </div>
                    </CardDescription>
                </CardHeader>
                <CardContent className="pb-0 pt-2">
                    <ResponsiveContainer width="100%" height={300}>
                        <ChartContainer config={chartConfig}>
                            <RadarChart data={chartData}>
                                <ChartTooltip
                                    cursor={false}
                                    content={
                                        <ChartTooltipContent
                                            className="text-gray-300 bg-customBlue"
                                        />
                                    }
                                />
                                <PolarAngleAxis dataKey="category" />
                                <PolarGrid />
                                <Radar
                                    name="FPTS"
                                    dataKey="value"
                                    stroke="hsl(var(--primary))"
                                    fill="hsl(var(--primary))"
                                    fillOpacity={0.6}
                                />
                            </RadarChart>
                        </ChartContainer>
                    </ResponsiveContainer>
                </CardContent>
                <div className='text-center'>
                    <p className="text-sm text-gray-500">Fantasy Points (FPTS) Distribution over Stat categories</p>
                </div>
            </>
        );

        return inDialog ? content : (
            <Card className="h-full card-border">
                {content}
            </Card>
        );
    }

    if (error) {
        return <div className="p-4 text-destructive">{error}</div>
    }

    return (
        <div className="container mx-auto p-4 space-y-6">
            <div className="flex justify-between items-center">
                <h1 className="text-gray-300 text-4xl font-bold">NBA Fantasy Dashboard</h1>
                <Button
                    className="fancy-button ghost-button"
                    onClick={handleResetAllData}
                >
                    <RefreshCcw className="mr-2 h-4 w-4" />
                    Reset All Data
                </Button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <Card className="lg:col-span-2 card-border flex flex-col">
                    <CardHeader>
                        <CardTitle className="text-gray-300">
                            Top {selectedPosition} Players
                        </CardTitle>
                        <CardDescription className="text-gray-300">
                            Fantasy Scores for 2024-2025 Season
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="flex-grow flex flex-col">
                        <div className="flex-grow">
                            <ChartContainer config={barChartConfig}>
                                <HorizontalBarChart data={topPlayers} />
                            </ChartContainer>
                        </div>
                        <PositionLegend />
                    </CardContent>
                    <CardFooter className="mt-auto">
                        <div className="flex gap-2 font-medium leading-none text-gray-300 text-mutedForeground">
                            Showing top players ({selectedPosition}) based on projected fantasy scores <TrendingUp className="h-4 w-4" />
                        </div>
                    </CardFooter>
                </Card>

                <Card className='card-border'>
                    <CardHeader>
                        <CardTitle className="text-gray-300 pb-2">Scoring Settings</CardTitle>
                        <CardDescription className="text-gray-300">Select a preset or customise to match your league's scoring system</CardDescription>
                    </CardHeader>
                    <CardContent>
                        <div className='pb-5 text-center'>
                            <Button
                                className="fancy-button ghost-button"
                                onClick={() => {
                                    applyPreset(espnPreset);
                                }}
                            >
                                ESPN
                            </Button>
                            <Button
                                className="fancy-button ghost-button"
                                onClick={() => {
                                    applyPreset(sleeperPreset);
                                }}
                            >
                                Sleeper
                            </Button>
                            <Button
                                className="fancy-button ghost-button"
                                onClick={() => {
                                    applyPreset(yahooPreset);
                                }}
                            >
                                Yahoo
                            </Button>
                            <Button
                                className="fancy-button ghost-button"
                                onClick={() => {
                                    applyPreset(fanduelPreset);
                                }}
                            >
                                FanDuel
                            </Button>
                            <Button
                                className="fancy-button ghost-button"
                                onClick={() => {
                                    applyPreset(nbaOfficialPreset);
                                }}
                            >
                                NBA
                            </Button>
                        </div>
                        <div className="grid grid-cols-2 gap-4">
                            {scoringCategories.map(category => (
                                <div key={category.name} className="space-y-2">
                                    <label className="text-gray-300 text-sm font-medium">{category.label}</label>
                                    <Slider
                                        value={[Math.abs(scoringSystem[category.name])]}
                                        max={5}
                                        step={0.1}
                                        onValueChange={(value) => handleScoringChange(category.name, category.defaultValue < 0 ? -value[0] : value[0])}
                                    />
                                    <span className="text-gray-300 text-sm text-muted-foreground">
                                        {getDisplayValue(category.name, scoringSystem[category.name])}
                                    </span>
                                </div>
                            ))}
                        </div>
                        <div className="mt-4 space-x-2">
                            <Button
                                className={`fancy-button ghost-button text-gray-500 border-gray-500 ${hasUnsavedChanges ? 'unsaved-changes' : ''}`}
                                onClick={handleUpdateScores}
                                disabled={isUpdating}
                            >
                                {isUpdating ? 'Updating...' : 'Update'}
                            </Button>
                            <Button className="fancy-button ghost-button text-gray-500 border-gray-500" onClick={() => {
                                const defaultSystem: ScoringSystem = {};
                                scoringCategories.forEach(category => {
                                    defaultSystem[category.name] = category.defaultValue;
                                });
                                setScoringSystem(defaultSystem);
                                handleUpdateScores();
                            }}>
                                Reset to Default
                            </Button>
                        </div>
                    </CardContent>
                </Card>
            </div>

            <Card className="card-border">
                <CardHeader>
                    <div className="pt-4 flex items-center space-x-2">
                        <Select value={selectedPosition} onValueChange={setSelectedPosition}>
                            <SelectTrigger className="text-gray-300 w-[180px]">
                                <SelectValue placeholder="Filter by position" />
                            </SelectTrigger>
                            <SelectContent className='bg-customBlue text-white'>
                                {positions.map(position => (
                                    <SelectItem className='hover:bg-blue-600' key={position} value={position}>{position}</SelectItem>
                                ))}
                            </SelectContent>
                        </Select>
                        <div className="relative flex-1">
                            <Search className="absolute left-2 top-2.5 h-4 w-4 text-gray-300 text-muted-foreground" />
                            <Input
                                placeholder="Search players..."
                                value={searchTerm}
                                onChange={(e) => setSearchTerm(e.target.value)}
                                className="pl-8 text-gray-300"
                            />
                        </div>
                    </div>
                </CardHeader>

                <CardContent className="max-h-[600px] overflow-auto custom-scrollbar">

                    <Table>
                        <TableHeader>
                            <TableRow className="text-gray-300">
                                <TableHead className="w-[50px]">Rank</TableHead>
                                <TableHead>Name</TableHead>
                                <TableHead>Position</TableHead>
                                <TableHead>MIN</TableHead>
                                <TableHead>PTS</TableHead>
                                <TableHead>AST</TableHead>
                                <TableHead>REB</TableHead>
                                <TableHead>STL</TableHead>
                                <TableHead>BLK</TableHead>
                                <TableHead>TOV</TableHead>
                                <TableHead>FGA</TableHead>
                                <TableHead>FGM</TableHead>
                                <TableHead>FTA</TableHead>
                                <TableHead>FTM</TableHead>
                                <TableHead>3PM</TableHead>
                                <TableHead className="text-center">Fantasy Points (FPTS)</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            {filteredPlayers.map((player) => (
                                <TableRow key={player.player_id} onClick={() => handlePlayerClick(player)} className="cursor-pointer">
                                    <TableCell className="text-gray-300 font-medium">{player.rank || 'N/A'}</TableCell>
                                    <TableCell className="text-gray-300">{player.player_name}</TableCell>
                                    <TableCell className="text-gray-300">{player.position || 'N/A'}</TableCell>
                                    <TableCell className="text-gray-300">{player.predicted_MP?.toFixed(1) || 'N/A'}</TableCell>
                                    <TableCell className="text-gray-300">{player.predicted_PTS?.toFixed(1) || 'N/A'}</TableCell>
                                    <TableCell className="text-gray-300">{player.predicted_AST?.toFixed(1) || 'N/A'}</TableCell>
                                    <TableCell className="text-gray-300">{(player.predicted_OREB + player.predicted_DREB)?.toFixed(1) || 'N/A'}</TableCell>
                                    <TableCell className="text-gray-300">{player.predicted_STL?.toFixed(1) || 'N/A'}</TableCell>
                                    <TableCell className="text-gray-300">{player.predicted_BLK?.toFixed(1) || 'N/A'}</TableCell>
                                    <TableCell className="text-gray-300">{player.predicted_TOV?.toFixed(1) || 'N/A'}</TableCell>
                                    <TableCell className="text-gray-300">{player.predicted_FGA?.toFixed(1) || 'N/A'}</TableCell>
                                    <TableCell className="text-gray-300">{player.predicted_FGM?.toFixed(1) || 'N/A'}</TableCell>
                                    <TableCell className="text-gray-300">{player.predicted_FTA?.toFixed(1) || 'N/A'}</TableCell>
                                    <TableCell className="text-gray-300">{player.predicted_FTM?.toFixed(1) || 'N/A'}</TableCell>
                                    <TableCell className="text-gray-300">{player.predicted_FG3M?.toFixed(1) || 'N/A'}</TableCell>
                                    <TableCell className="text-center text-gray-300">
                                        {player.fantasy_score !== undefined ? player.fantasy_score.toFixed(2) : 'N/A'}
                                    </TableCell>
                                    <TableCell>
                                        <Button
                                            className="ghost-button"
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                handleRemovePlayer(player.player_id);
                                            }}
                                        >
                                            <X className="h-4 w-4 text-blue-400" />
                                        </Button>
                                    </TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>

                </CardContent>
            </Card>

            <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
                <DialogContent className="bg-customBlue text-gray-300">
                    {renderPlayerStats(true)}
                </DialogContent>
            </Dialog>
        </div>
    )
}
