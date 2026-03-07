import 'package:flutter/material.dart';
import '../theme/app_theme.dart';
import '../services/api_service.dart';
import '../models/prediction.dart';
import '../models/live_match.dart';
import '../widgets/glass_card.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  List<TeamRanking>? _rankings;
  List<LiveMatch>? _liveMatches;
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _loadAll();
  }

  Future<void> _loadAll() async {
    setState(() { _loading = true; _error = null; });
    try {
      final results = await Future.wait([
        ApiService.getTournamentPredictions(),
        ApiService.getLiveMatches(),
      ]);
      setState(() {
        _rankings = results[0] as List<TeamRanking>;
        _liveMatches = results[1] as List<LiveMatch>;
        _loading = false;
      });
    } catch (e) {
      setState(() { _error = e.toString(); _loading = false; });
    }
  }

  Future<void> _loadRankings() async {
    await _loadAll();
  }

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: RefreshIndicator(
        onRefresh: _loadRankings,
        color: AppTheme.accent,
        child: CustomScrollView(
          slivers: [
            // Header
            SliverToBoxAdapter(
              child: Padding(
                padding: const EdgeInsets.fromLTRB(20, 24, 20, 8),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('IPL Predictor', style: Theme.of(context).textTheme.headlineLarge),
                    const SizedBox(height: 4),
                    Text('AI-powered match predictions', style: Theme.of(context).textTheme.bodyMedium),
                  ],
                ),
              ),
            ),

            // Hero card
            SliverToBoxAdapter(
              child: Padding(
                padding: const EdgeInsets.all(20),
                child: Container(
                  padding: const EdgeInsets.all(24),
                  decoration: BoxDecoration(
                    gradient: AppTheme.heroGradient,
                    borderRadius: BorderRadius.circular(20),
                    boxShadow: [
                      BoxShadow(
                        color: AppTheme.accent.withOpacity(0.2),
                        blurRadius: 20,
                        offset: const Offset(0, 8),
                      ),
                    ],
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Icon(Icons.auto_awesome, color: Colors.white, size: 32),
                      const SizedBox(height: 12),
                      Text('Ensemble AI Model', style: Theme.of(context).textTheme.titleLarge?.copyWith(color: Colors.white)),
                      const SizedBox(height: 4),
                      Text(
                        'XGBoost + LightGBM + Logistic Regression with SHAP explanations',
                        style: Theme.of(context).textTheme.bodyMedium?.copyWith(color: Colors.white70),
                      ),
                    ],
                  ),
                ),
              ),
            ),

            // Live match cards (shown only when matches exist)
            if (!_loading && _liveMatches != null && _liveMatches!.isNotEmpty)
              SliverToBoxAdapter(
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(20, 0, 20, 8),
                  child: Row(
                    children: [
                      Text('Live Matches', style: Theme.of(context).textTheme.titleLarge),
                      const SizedBox(width: 8),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                        decoration: BoxDecoration(
                          color: AppTheme.coral.withOpacity(0.15),
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Text(
                          '${_liveMatches!.length}',
                          style: const TextStyle(color: AppTheme.coral, fontWeight: FontWeight.w700, fontSize: 12),
                        ),
                      ),
                    ],
                  ),
                ),
              ),

            if (!_loading && _liveMatches != null && _liveMatches!.isNotEmpty)
              SliverList(
                delegate: SliverChildBuilderDelegate(
                  (context, index) {
                    final match = _liveMatches![index];
                    return Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 4),
                      child: _buildLiveMatchCard(context, match),
                    );
                  },
                  childCount: _liveMatches!.length,
                ),
              ),

            if (!_loading && _liveMatches != null && _liveMatches!.isNotEmpty)
              const SliverToBoxAdapter(child: SizedBox(height: 12)),

            // Tournament Rankings header
            SliverToBoxAdapter(
              child: Padding(
                padding: const EdgeInsets.fromLTRB(20, 8, 20, 12),
                child: Text('Tournament Win Probability', style: Theme.of(context).textTheme.titleLarge),
              ),
            ),

            // Rankings list
            if (_loading)
              const SliverFillRemaining(child: Center(child: CircularProgressIndicator(color: AppTheme.accent)))
            else if (_error != null)
              SliverFillRemaining(
                child: Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const Icon(Icons.cloud_off, size: 48, color: AppTheme.textSecondary),
                      const SizedBox(height: 12),
                      Text('Could not load rankings', style: Theme.of(context).textTheme.bodyLarge),
                      const SizedBox(height: 8),
                      TextButton(onPressed: _loadRankings, child: const Text('Retry')),
                    ],
                  ),
                ),
              )
            else
              SliverList(
                delegate: SliverChildBuilderDelegate(
                  (context, index) {
                    final team = _rankings![index];
                    final teamColor = AppTheme.getTeamColor(team.team);
                    return Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 4),
                      child: GlassCard(
                        child: Row(
                          children: [
                            // Rank badge
                            Container(
                              width: 36, height: 36,
                              decoration: BoxDecoration(
                                color: index < 3 ? teamColor.withOpacity(0.2) : AppTheme.surface,
                                borderRadius: BorderRadius.circular(10),
                              ),
                              child: Center(
                                child: Text(
                                  '${team.rank}',
                                  style: TextStyle(
                                    color: index < 3 ? teamColor : AppTheme.textSecondary,
                                    fontWeight: FontWeight.w700,
                                    fontSize: 16,
                                  ),
                                ),
                              ),
                            ),
                            const SizedBox(width: 14),
                            // Team colour dot
                            Container(width: 10, height: 10, decoration: BoxDecoration(color: teamColor, shape: BoxShape.circle)),
                            const SizedBox(width: 10),
                            // Team name
                            Expanded(child: Text(team.team, style: Theme.of(context).textTheme.titleMedium)),
                            // Probability
                            Text(
                              '${(team.winProbability * 100).toStringAsFixed(1)}%',
                              style: Theme.of(context).textTheme.titleMedium?.copyWith(
                                color: teamColor,
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                          ],
                        ),
                      ),
                    );
                  },
                  childCount: _rankings?.length ?? 0,
                ),
              ),
            const SliverToBoxAdapter(child: SizedBox(height: 20)),
          ],
        ),
      ),
    );
  }

  Widget _buildLiveMatchCard(BuildContext context, LiveMatch match) {
    final team1Color = AppTheme.getTeamColor(match.team1);
    final team2Color = AppTheme.getTeamColor(match.team2);
    final isLive = match.isLive;

    return GlassCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Status row
          Row(
            children: [
              if (isLive)
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                  decoration: BoxDecoration(
                    color: AppTheme.coral.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(6),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Container(
                        width: 6, height: 6,
                        decoration: const BoxDecoration(color: AppTheme.coral, shape: BoxShape.circle),
                      ),
                      const SizedBox(width: 4),
                      const Text('LIVE', style: TextStyle(color: AppTheme.coral, fontWeight: FontWeight.w700, fontSize: 11)),
                    ],
                  ),
                )
              else
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                  decoration: BoxDecoration(
                    color: AppTheme.accent.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(6),
                  ),
                  child: Text(
                    match.matchEnded ? 'Completed' : 'Upcoming',
                    style: TextStyle(color: AppTheme.accent, fontWeight: FontWeight.w600, fontSize: 11),
                  ),
                ),
              const Spacer(),
              if (match.venue.isNotEmpty)
                Flexible(
                  child: Text(
                    match.venue,
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(color: AppTheme.textSecondary),
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
            ],
          ),
          const SizedBox(height: 12),

          // Teams and scores
          Row(
            children: [
              Container(width: 10, height: 10, decoration: BoxDecoration(color: team1Color, shape: BoxShape.circle)),
              const SizedBox(width: 8),
              Expanded(
                child: Text(match.team1, style: Theme.of(context).textTheme.titleMedium),
              ),
              if (match.team1Score.isNotEmpty)
                Text(match.team1Score, style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700)),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              Container(width: 10, height: 10, decoration: BoxDecoration(color: team2Color, shape: BoxShape.circle)),
              const SizedBox(width: 8),
              Expanded(
                child: Text(match.team2, style: Theme.of(context).textTheme.titleMedium),
              ),
              if (match.team2Score.isNotEmpty)
                Text(match.team2Score, style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700)),
            ],
          ),

          // Win probability bar (if live prediction available)
          if (match.liveProbTeam1 != null) ...[
            const SizedBox(height: 14),
            Row(
              children: [
                Text(
                  '${(match.liveProbTeam1! * 100).toStringAsFixed(0)}%',
                  style: TextStyle(color: team1Color, fontWeight: FontWeight.w700, fontSize: 13),
                ),
                const Spacer(),
                Text(
                  '${(match.liveProbTeam2! * 100).toStringAsFixed(0)}%',
                  style: TextStyle(color: team2Color, fontWeight: FontWeight.w700, fontSize: 13),
                ),
              ],
            ),
            const SizedBox(height: 4),
            ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: SizedBox(
                height: 6,
                child: Row(
                  children: [
                    Flexible(
                      flex: (match.liveProbTeam1! * 100).round(),
                      child: Container(color: team1Color),
                    ),
                    Flexible(
                      flex: (match.liveProbTeam2! * 100).round(),
                      child: Container(color: team2Color),
                    ),
                  ],
                ),
              ),
            ),
          ],

          // Status text
          if (match.status.isNotEmpty) ...[
            const SizedBox(height: 10),
            Text(
              match.status,
              style: Theme.of(context).textTheme.bodySmall?.copyWith(color: AppTheme.textSecondary),
            ),
          ],
        ],
      ),
    );
  }
}
