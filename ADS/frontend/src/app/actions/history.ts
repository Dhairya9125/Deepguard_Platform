'use server';

import { db } from '@/db';
import { analysisJobs, forensicResults } from '@/db/schema';
import { desc, eq } from 'drizzle-orm';

export async function getHistoryData() {
  try {
    const data = await db
      .select()
      .from(analysisJobs)
      .leftJoin(forensicResults, eq(analysisJobs.id, forensicResults.jobId))
      .orderBy(desc(analysisJobs.createdAt));

    return data.map((row) => {
      const job = row.analysis_jobs;
      const result = row.forensic_results;
      
      let statusText = 'Review Needed';
      let score = 'N/A';
      
      if (job.status === 'COMPLETED' && result) {
        statusText = result.verdict === 'FAKE' ? 'Detected' : (result.verdict === 'REAL' ? 'Authentic' : 'Uncertain');
        score = `${(result.fakeProbability * 100).toFixed(0)}%`;
      } else if (job.status === 'PENDING' || job.status === 'PROCESSING') {
        statusText = 'Processing';
      } else if (job.status === 'FAILED') {
        statusText = 'Failed';
      }

      return {
        id: job.id,
        file: job.fileName || 'Unknown File',
        type: job.modality.charAt(0).toUpperCase() + job.modality.slice(1),
        date: job.createdAt.toISOString().split('T')[0],
        status: statusText,
        score: score,
      };
    });
  } catch (error) {
    console.error('Failed to fetch history:', error);
    return [];
  }
}
