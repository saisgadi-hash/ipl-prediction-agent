import 'package:flutter/material.dart';
import '../theme/app_theme.dart';
import '../services/api_service.dart';
import '../models/prediction.dart';
import '../widgets/glass_card.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  List<TeamRanking>? _rankings;
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _loadRankings();
  }

  Future<void> _loadRankings() async {
    setState(() { _loading = true; _error = null; });
    try {
      final rankings = await ApiService.getTournamentPredictions();
      setState(() { _rankings = rankings; _loading = false; });
    } catch (e) {
      setState(() { _error = e.toString(); _loading = false; });
    }
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
                    gradient: AppTheme.accentGradient,
                    borderRadius: BorderRadius.circular(20),
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
}
