import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import '../theme/app_theme.dart';
import '../services/api_service.dart';
import '../models/prediction.dart';
import '../widgets/glass_card.dart';

class TournamentScreen extends StatefulWidget {
  const TournamentScreen({super.key});

  @override
  State<TournamentScreen> createState() => _TournamentScreenState();
}

class _TournamentScreenState extends State<TournamentScreen> {
  List<TeamRanking>? _rankings;
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() { _loading = true; _error = null; });
    try {
      final rankings = await ApiService.getTournamentPredictions();
      setState(() { _rankings = rankings; _loading = false; });
    } catch (e) {
      setState(() { _error = e.toString(); _loading = false; });
    }
  }

  String _formIcon(String state) {
    switch (state) {
      case 'Hot': return '🔥';
      case 'Cold': return '❄️';
      default: return '⚡';
    }
  }

  Color _formColor(String state) {
    switch (state) {
      case 'Hot': return AppTheme.mint;
      case 'Cold': return AppTheme.coral;
      default: return AppTheme.golden;
    }
  }

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: RefreshIndicator(
        onRefresh: _load,
        color: AppTheme.accent,
        child: SingleChildScrollView(
          physics: const AlwaysScrollableScrollPhysics(),
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('Tournament', style: Theme.of(context).textTheme.headlineLarge),
              const SizedBox(height: 4),
              Text('Monte Carlo simulation — 5,000 season simulations', style: Theme.of(context).textTheme.bodyMedium),
              const SizedBox(height: 24),

              if (_loading)
                const SizedBox(height: 300, child: Center(child: CircularProgressIndicator(color: AppTheme.accent)))
              else if (_error != null)
                Center(child: Column(
                  children: [
                    const Icon(Icons.cloud_off, size: 48, color: AppTheme.textSecondary),
                    const SizedBox(height: 12),
                    Text('Could not load tournament data', style: Theme.of(context).textTheme.bodyLarge),
                    TextButton(onPressed: _load, child: const Text('Retry')),
                  ],
                ))
              else ...[
                // Bar chart
                GlassCard(
                  child: SizedBox(
                    height: 300,
                    child: BarChart(
                      BarChartData(
                        alignment: BarChartAlignment.spaceAround,
                        maxY: (_rankings!.first.winProbability * 100) * 1.2,
                        barGroups: _rankings!.map((t) {
                          final color = AppTheme.getTeamColor(t.team);
                          return BarChartGroupData(
                            x: t.rank,
                            barRods: [
                              BarChartRodData(
                                toY: t.winProbability * 100,
                                color: color,
                                width: 18,
                                borderRadius: const BorderRadius.vertical(top: Radius.circular(6)),
                              ),
                            ],
                          );
                        }).toList(),
                        titlesData: FlTitlesData(
                          leftTitles: AxisTitles(
                            sideTitles: SideTitles(
                              showTitles: true,
                              reservedSize: 40,
                              getTitlesWidget: (value, meta) => Text('${value.toInt()}%', style: const TextStyle(color: AppTheme.textSecondary, fontSize: 11)),
                            ),
                          ),
                          bottomTitles: AxisTitles(
                            sideTitles: SideTitles(
                              showTitles: true,
                              getTitlesWidget: (value, meta) {
                                final idx = value.toInt() - 1;
                                if (idx < 0 || idx >= _rankings!.length) return const Text('');
                                final name = _rankings![idx].team;
                                final abbr = name.split(' ').map((w) => w[0]).join('');
                                return Padding(
                                  padding: const EdgeInsets.only(top: 6),
                                  child: Text(abbr, style: TextStyle(color: AppTheme.getTeamColor(name), fontSize: 10, fontWeight: FontWeight.w600)),
                                );
                              },
                            ),
                          ),
                          topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                          rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                        ),
                        gridData: FlGridData(
                          show: true,
                          drawVerticalLine: false,
                          horizontalInterval: 5,
                          getDrawingHorizontalLine: (value) => FlLine(color: AppTheme.border, strokeWidth: 0.5),
                        ),
                        borderData: FlBorderData(show: false),
                      ),
                    ),
                  ),
                ),

                const SizedBox(height: 20),

                // Team Form States
                Text('Team Form', style: Theme.of(context).textTheme.titleLarge),
                const SizedBox(height: 4),
                Text('HMM-detected form state', style: Theme.of(context).textTheme.bodySmall),
                const SizedBox(height: 12),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: _rankings!.map((t) {
                    final color = AppTheme.getTeamColor(t.team);
                    final abbr = t.team.split(' ').map((w) => w[0]).join('');
                    return Container(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                      decoration: BoxDecoration(
                        color: _formColor(t.formState).withOpacity(0.1),
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(color: _formColor(t.formState).withOpacity(0.3)),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Text(_formIcon(t.formState), style: const TextStyle(fontSize: 16)),
                          const SizedBox(width: 6),
                          Text(abbr, style: TextStyle(color: color, fontWeight: FontWeight.w700, fontSize: 13)),
                        ],
                      ),
                    );
                  }).toList(),
                ),

                const SizedBox(height: 24),

                // Detailed rankings with Elo
                Text('Detailed Rankings', style: Theme.of(context).textTheme.titleLarge),
                const SizedBox(height: 12),
                ..._rankings!.map((t) {
                  final color = AppTheme.getTeamColor(t.team);
                  final pct = (t.winProbability * 100);
                  return Padding(
                    padding: const EdgeInsets.only(bottom: 8),
                    child: GlassCard(
                      child: Column(
                        children: [
                          Row(children: [
                            // Rank
                            SizedBox(
                              width: 30,
                              child: Text('#${t.rank}', style: TextStyle(color: color, fontWeight: FontWeight.w700, fontSize: 16)),
                            ),
                            Container(width: 10, height: 10, decoration: BoxDecoration(color: color, shape: BoxShape.circle)),
                            const SizedBox(width: 10),
                            Expanded(child: Text(t.team, style: Theme.of(context).textTheme.titleMedium)),
                            Text(_formIcon(t.formState), style: const TextStyle(fontSize: 14)),
                            const SizedBox(width: 8),
                            SizedBox(
                              width: 48,
                              child: Text('${pct.toStringAsFixed(1)}%', textAlign: TextAlign.right,
                                style: TextStyle(color: color, fontWeight: FontWeight.w700)),
                            ),
                          ]),
                          const SizedBox(height: 8),
                          Row(children: [
                            const SizedBox(width: 40),
                            // Progress bar
                            Expanded(
                              child: ClipRRect(
                                borderRadius: BorderRadius.circular(4),
                                child: LinearProgressIndicator(
                                  value: t.winProbability,
                                  backgroundColor: AppTheme.surface,
                                  color: color,
                                  minHeight: 6,
                                ),
                              ),
                            ),
                            const SizedBox(width: 12),
                            Text('Elo ${t.eloRating.toStringAsFixed(0)}',
                              style: Theme.of(context).textTheme.bodySmall?.copyWith(color: AppTheme.textSecondary)),
                          ]),
                        ],
                      ),
                    ),
                  );
                }),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
